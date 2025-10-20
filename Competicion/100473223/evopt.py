# Imports
import numpy as np
import pandas as pd
import time
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler

import warnings
warnings.filterwarnings('ignore')

# Estructura de árboles
class GPNode:
    """Creación manual de un nodo de árbol de programación genética."""
    
    def __init__(self, value, children=None, node_type='terminal'):
        self.value = value
        self.children = children or []
        self.node_type = node_type
    
    def evaluate(self, X):
        """Evalúa el nodo con los datos X."""
        if self.node_type == 'terminal':
            if isinstance(self.value, int):
                return X[:, self.value]
            else:
                return np.full(X.shape[0], self.value)
        else:
            if self.value == 'add':
                return self.children[0].evaluate(X) + self.children[1].evaluate(X)
            elif self.value == 'sub':
                return self.children[0].evaluate(X) - self.children[1].evaluate(X)
            elif self.value == 'mul':
                return self.children[0].evaluate(X) * self.children[1].evaluate(X)
            elif self.value == 'div':
                right = self.children[1].evaluate(X)
                return self.children[0].evaluate(X) / (right + 1e-6)
            elif self.value == 'sqrt':
                return np.sqrt(np.abs(self.children[0].evaluate(X)) + 1e-6)
            elif self.value == 'square':
                val = self.children[0].evaluate(X)
                return np.clip(val, -100, 100) ** 2 # We clip the input to avoid overflow
            elif self.value == 'log':
                return np.log(np.abs(self.children[0].evaluate(X)) + 1)
            elif self.value == 'sin':
                return np.sin(self.children[0].evaluate(X))
            elif self.value == 'cos':
                return np.cos(self.children[0].evaluate(X))
            elif self.value == 'tanh':
                return np.tanh(self.children[0].evaluate(X))
            elif self.value == 'sigmoid':
                return 1 / (1 + np.exp(-self.children[0].evaluate(X)))
    
    def copy(self):
        """Copia profunda del nodo."""
        return GPNode(
            self.value,
            [child.copy() for child in self.children],
            self.node_type
        )
    
    def size(self):
        """Tamaño del árbol (número de nodos)."""
        return 1 + sum(child.size() for child in self.children)
    
    def depth(self):
        """Profundidad del árbol."""
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)
    
    def to_string(self):
        """Representación en string del árbol."""
        if self.node_type == 'terminal':
            if isinstance(self.value, int):
                return f"X{self.value}"
            else:
                return f"{self.value:.3f}"
        else:
            if len(self.children) == 1:
                return f"{self.value}({self.children[0].to_string()})"
            else:
                return f"({self.children[0].to_string()} {self.value} {self.children[1].to_string()})"


class EvolutionaryOptimizer(BaseEstimator, TransformerMixin):
    """Optimizador de programación genética para crear features usando Cross-Validation completo."""
    
    def __init__(self, maxtime=1200, population_size=60, n_features_to_create=4,
                 mutation_prob=0.15, crossover_prob=0.8, tournament_size=3,
                 max_depth=4, elite_size=0.15, apply_feature_selection=True,
                 evaluation_model='ensemble', cv_folds=3, random_state=100473223):
        self.maxtime = maxtime
        self.population_size = population_size
        self.n_features_to_create = n_features_to_create
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        self.elite_size = int(population_size * elite_size)
        self.apply_feature_selection = apply_feature_selection
        self.evaluation_model = evaluation_model
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Funciones disponibles - REDUCIDAS PARA EVITAR OVERFITTING
        self.functions = {
            'add': 2, 'sub': 2, 'mul': 2, 'div': 2,
            'sqrt': 1, 'square': 1, 'log': 1, 'tanh': 1 #sin, cos, 'sigmoid' removed
        }
        
        self.best_trees_ = []
        self.best_fitness_ = float('inf')
        self.fitness_history_ = []
        self.best_metrics_ = {'mae': None, 'mse': None}
        self.feature_selection_ = None
        self.cv_strategy_ = None
        
        # Variables de control de tiempo
        self._start_time = None
        self._gp_deadline = None
        self._fs_deadline = None
        self._total_deadline = None
    ##
    # Funciones adicionales para evitar que se exceda el tiempo en medio de 1 iteración
    ##
    def _check_time_remaining(self, phase="general"): 
        """Verifica si queda tiempo disponible para continuar."""
        current_time = time.time()
        
        if phase == "gp":
            return current_time < self._gp_deadline
        elif phase == "fs":
            return current_time < self._fs_deadline
        else:
            return current_time < self._total_deadline
    
    def _get_time_elapsed(self):
        """Retorna el tiempo transcurrido desde el inicio."""
        return time.time() - self._start_time
    
    def _get_time_remaining(self, phase="total"):
        """Retorna el tiempo restante para una fase específica."""
        current_time = time.time()
        
        if phase == "gp":
            return max(0, self._gp_deadline - current_time)
        elif phase == "fs":
            return max(0, self._fs_deadline - current_time)
        else:
            return max(0, self._total_deadline - current_time)
    
    def fit(self, X, y):
        """Entrena usando programación genética y selección de features con CV completo."""
        # Inicializar control de tiempo estricto
        self._start_time = time.time()
        
        # División del tiempo: 70% GP, 20% FS, 10% buffer de seguridad
        gp_time = self.maxtime * 0.70
        fs_time = self.maxtime * 0.20
        buffer_time = self.maxtime * 0.10 # 'Por si acaso'
        
        # Establecer deadlines
        self._gp_deadline = self._start_time + gp_time
        self._fs_deadline = self._start_time + gp_time + fs_time
        self._total_deadline = self._start_time + self.maxtime - buffer_time
        
        # Fijar semilla si se especifica
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)
        
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Asegurar que y sea 1D
        if len(y.shape) > 1: # Intento de prevenir Warning en validator.py
            y = y.ravel()
        
        # Configurar estrategia de CV consistente
        self.cv_strategy_ = KFold(
            n_splits=self.cv_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.n_features_in_ = X.shape[1]
        print("Juan Leal Aliaga - 100473223")
        print(f"\n{'='*70}")
        print(f"PROGRAMACIÓN GENÉTICA CON CONTROL ESTRICTO DE TIEMPO")
        #print(f"{'='*70}")
        #print(f"Tiempo total asignado: {self.maxtime}s ({self.maxtime/60:.1f}min)")
        #print(f"  - Programación Genética: {gp_time}s ({gp_time/60:.1f}min) - 70%")
        #print(f"  - Feature Selection: {fs_time}s ({fs_time/60:.1f}min) - 20%")
        #print(f"  - Buffer de seguridad: {buffer_time}s ({buffer_time/60:.1f}min) - 10%")
        #print(f"Población: {self.population_size} | Features a crear: {self.n_features_to_create}")
        #print(f"CV Folds: {self.cv_folds} | Profundidad máxima: {self.max_depth}")
        print(f"{'='*70}\n")
        
        # FASE 1: PROGRAMACIÓN GENÉTICA CON CONTROL ESTRICTO DE TIEMPO
        success_gp = self._run_genetic_programming(X_scaled, y, gp_time)
        
        if not success_gp:
            print(f"Programación Genética interrumpida por límite de tiempo")
        
        # Verificar si queda tiempo para Feature Selection
        if self.apply_feature_selection and self._check_time_remaining("fs"):
            remaining_fs_time = self._get_time_remaining("fs")
            #print(f"\n{'='*70}")
            print(f"FEATURE SELECTION CON CONTROL ESTRICTO DE TIEMPO")
            print(f"{'='*70}")
            #print(f"Tiempo disponible para FS: {remaining_fs_time:.1f}s ({remaining_fs_time/60:.1f}min)")
            
            success_fs = self._run_feature_selection(X, y, remaining_fs_time)
            
            if not success_fs:
                print(f"⚠️  Feature Selection interrumpida por límite de tiempo")
        elif self.apply_feature_selection:
            print(f"\n⚠️  No hay tiempo suficiente para Feature Selection")
            print(f"Tiempo restante: {self._get_time_remaining():.1f}s")
        
        total_elapsed = self._get_time_elapsed()
        efficiency = (total_elapsed / self.maxtime) * 100
        
        #print(f"\n{'='*70}")
        print(f"ENTRENAMIENTO COMPLETADO")
        print(f"{'='*70}")
        #print(f"Tiempo usado: {total_elapsed:.1f}s de {self.maxtime}s ({efficiency:.1f}%)")
        #print(f"Tiempo restante: {self._get_time_remaining():.1f}s")
        #print(f"Estado: {'✓ DENTRO DEL LÍMITE' if total_elapsed < self.maxtime else '⚠️ EXCEDIDO'}")
        #print(f"{'='*70}")

        return self
    
    def _run_genetic_programming(self, X_scaled, y, allocated_time):
        """Ejecuta la programación genética con control estricto de tiempo."""
        print(f"Iniciando Programación Genética...")
        gp_start = time.time()
        
        # Inicializar población
        population = [self._create_random_tree() for _ in range(self.population_size)]
        
        generation = 0
        gp_early_stop = 0
        best_cv_fitness = float('inf')
        
        while self._check_time_remaining("gp"):
            generation_start = time.time()
            generation += 1
            
            # Verificar si hay tiempo suficiente para completar una generación
            if self._get_time_remaining("gp") < 10:  # Buffer mínimo de 10s por generación
                print(f"  Tiempo insuficiente para completar generación {generation}")
                break
            
            # Evaluar población con timeout
            cv_fitness, cv_metrics_list = self._evaluate_population_with_timeout(
                population, X_scaled, y, self._get_time_remaining("gp") * 0.8
            )
            
            if cv_fitness is None:  # Timeout en evaluación
                print(f"  Timeout en evaluación de generación {generation}")
                break
            
            # Actualizar mejor
            best_cv_idx = np.argmin(cv_fitness)
            if cv_fitness[best_cv_idx] < best_cv_fitness:
                best_cv_fitness = cv_fitness[best_cv_idx]
                self.best_fitness_ = best_cv_fitness
                self.best_trees_ = [tree.copy() for tree in population[best_cv_idx]]
                self.best_metrics_ = cv_metrics_list[best_cv_idx]
                gp_early_stop = 0
                print(f"Gen {generation} - MEJORA! CV MSE: {best_cv_fitness:.4f} | " +
                      f"CV MAE: {self.best_metrics_['mae']:.4f} | " +
                      f"Tiempo: {self._get_time_elapsed():.1f}s")
            else:
                gp_early_stop += 1
            
            # Early stopping
            if gp_early_stop >= 30:
                print(f"  GP Early stopping en generación {generation}")
                break
            
            self.fitness_history_.append(self.best_fitness_)
            
            # Verificar tiempo antes de crear nueva generación
            if not self._check_time_remaining("gp"):
                break
            
            # Crear nueva generación con timeout
            population = self._create_new_generation_with_timeout(
                population, cv_fitness, self._get_time_remaining("gp") * 0.8
            )
            
            if population is None:  # Timeout en creación de nueva generación
                print(f"  Timeout en creación de nueva generación {generation}")
                break
            
            # Log cada 20 generaciones
            if generation % 20 == 0:
                elapsed = self._get_time_elapsed()
                remaining = self._get_time_remaining("gp")
                print(f"Gen {generation} | CV MSE: {best_cv_fitness:.4f} | " +
                      f"Tiempo: {elapsed:.1f}s | Restante GP: {remaining:.1f}s")
        
        gp_elapsed = time.time() - gp_start
        print(f"Programación Genética completada: {generation} generaciones en {gp_elapsed:.1f}s")
        
        if self.best_trees_:
            print(f"Mejores árboles encontrados:")
            for i, tree in enumerate(self.best_trees_):
                print(f"  {i+1}: {tree.to_string()}")
        
        return generation > 0
    
    def _evaluate_population_with_timeout(self, population, X, y, timeout):
        """Evalúa población con timeout."""
        eval_start = time.time()
        cv_fitness = []
        cv_metrics_list = []
        
        for i, individual in enumerate(population):
            # Verificar timeout
            if (time.time() - eval_start) > timeout:
                print(f"    Timeout en evaluación individual {i}/{len(population)}")
                return None, None
            
            cv_fit, cv_metrics = self._evaluate_individual_cv(individual, X, y)
            cv_fitness.append(cv_fit)
            cv_metrics_list.append(cv_metrics)
        
        return cv_fitness, cv_metrics_list
    
    def _create_new_generation_with_timeout(self, population, cv_fitness, timeout):
        """Crea nueva generación con timeout."""
        gen_start = time.time()
        new_population = []
        
        # Elitismo
        elite_indices = np.argsort(cv_fitness)[:self.elite_size]
        for idx in elite_indices:
            new_population.append([tree.copy() for tree in population[idx]])
        
        # Generar resto con timeout
        while len(new_population) < self.population_size:
            # Verificar timeout
            if (time.time() - gen_start) > timeout:
                print(f"    Timeout en creación de nueva generación")
                return None
            
            # Selección por torneo
            parent1 = self._tournament_selection(population, cv_fitness)
            parent2 = self._tournament_selection(population, cv_fitness)
            
            # Cruce y mutación
            if random.random() < self.crossover_prob:
                child1, child2 = self._crossover_trees(parent1, parent2)
            else:
                child1 = [tree.copy() for tree in parent1]
                child2 = [tree.copy() for tree in parent2]
            
            if random.random() < self.mutation_prob:
                child1 = self._mutate_trees(child1)
            if random.random() < self.mutation_prob:
                child2 = self._mutate_trees(child2)
            
            new_population.extend([child1, child2])
        
        return new_population[:self.population_size]
    
    def _run_feature_selection(self, X, y, allocated_time):
        """Ejecuta feature selection con control estricto de tiempo."""
        # Transformar dataset
        X_transformed = self._transform_without_selection(X)
        
        print(f"Aplicando selección evolutiva sobre {X_transformed.shape[1]} features...")
        
        fs_start = time.time()
        
        # Ejecutar con timeout estricto
        self.feature_selection_, fs_metrics = self._evolutionary_feature_selection_with_timeout(
            X_transformed, y, allocated_time * 0.9  # 90% del tiempo asignado
        )
        
        fs_elapsed = time.time() - fs_start
        
        if self.feature_selection_ is not None:
            n_selected = np.sum(self.feature_selection_)
            print(f"✓ Selección completada: {n_selected}/{len(self.feature_selection_)} features")
            print(f"  Mejor CV MAE: {fs_metrics['mae']:.4f}")
            print(f"  Mejor CV MSE: {fs_metrics['mse']:.4f}")
            print(f"  Tiempo usado: {fs_elapsed:.1f}s")
            return True
        else:
            print(f"⚠️ Feature Selection no completada en tiempo asignado")
            return False
    
    def _evolutionary_feature_selection_with_timeout(self, X_full, y_full, max_time):
        """Feature selection con timeout estricto."""
        n_features = X_full.shape[1]
        population_size = 60  # ¿Valor óptimo? Difícil de balancear (Rendimiento en validator, mi benchmark, overfitting,...)
        
        # Inicializar población
        population = []
        for _ in range(population_size):
            individual = np.zeros(n_features, dtype=bool)
            n_selected = random.randint(3, min(12, n_features))
            selected_idx = random.sample(range(n_features), n_selected)
            individual[selected_idx] = True
            population.append(individual)
        
        best_individual = None
        best_metrics = {'mae': float('inf'), 'mse': float('inf')}
        best_cv_fitness = float('inf')
        
        fs_start = time.time()
        gen = 0
        
        while (time.time() - fs_start) < max_time:
            gen += 1
            
            # Verificar tiempo restante
            remaining_time = max_time - (time.time() - fs_start)
            if remaining_time < 5:  # Buffer mínimo
                break
            
            # Evaluar población con timeout
            cv_fitness, cv_metrics_list = self._evaluate_fs_population_with_timeout(
                population, X_full, y_full, remaining_time * 0.8
            )
            
            if cv_fitness is None:
                break
            
            # Actualizar mejor
            current_best_idx = np.argmin(cv_fitness)
            if cv_fitness[current_best_idx] < best_cv_fitness:
                best_cv_fitness = cv_fitness[current_best_idx]
                best_individual = population[current_best_idx].copy()
                best_metrics = cv_metrics_list[current_best_idx]
                
                if gen % 5 == 0:
                    n_selected = np.sum(best_individual)
                    elapsed = time.time() - fs_start
                    print(f"  Gen {gen}: CV MSE = {best_metrics['mse']:.4f} | " +
                          f"Features: {n_selected} | Tiempo: {elapsed:.1f}s")
            
            # Verificar tiempo antes de nueva generación
            if (time.time() - fs_start) >= max_time * 0.95:
                break
            
            # Crear nueva generación
            population = self._create_new_fs_generation(population, cv_fitness)
        
        return best_individual, best_metrics
    
    def _evaluate_fs_population_with_timeout(self, population, X_full, y_full, timeout):
        """Evalúa población de FS con timeout."""
        eval_start = time.time()
        cv_fitness = []
        cv_metrics_list = []
        
        for i, individual in enumerate(population):
            if (time.time() - eval_start) > timeout:
                return None, None
            
            cv_fit, cv_metrics = self._evaluate_feature_subset_cv(individual, X_full, y_full)
            cv_fitness.append(cv_fit)
            cv_metrics_list.append(cv_metrics)
        
        return cv_fitness, cv_metrics_list
    
    def _create_new_fs_generation(self, population, cv_fitness):
        """Crea nueva generación para FS."""
        new_population = []
        population_size = len(population)
        
        # Elitismo
        elite_size = max(1, population_size // 10)
        elite_indices = np.argsort(cv_fitness)[:elite_size]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generar resto
        while len(new_population) < population_size:
            parent1 = self._tournament_selection_fs(population, cv_fitness, 3)
            parent2 = self._tournament_selection_fs(population, cv_fitness, 3)
            
            if random.random() < 0.7:
                child1, child2 = self._crossover_fs(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            if random.random() < 0.2:
                child1 = self._mutate_fs(child1)
            if random.random() < 0.2:
                child2 = self._mutate_fs(child2)
            
            new_population.extend([child1, child2])
        
        return new_population[:population_size]
    
    def _evaluate_individual_cv(self, trees, X, y):
        """Evalúa un individuo usando Cross-Validation con ensemble."""
        try:
            X_new = X.copy()
            for tree in trees:
                try:
                    new_feature = tree.evaluate(X)
                    new_feature = np.nan_to_num(new_feature, nan=0.0, posinf=100.0, neginf=-100.0)
                    new_feature = np.clip(new_feature, -1000, 1000)
                    X_new = np.column_stack([X_new, new_feature])
                except:
                    X_new = np.column_stack([X_new, np.zeros(X.shape[0])])

            if np.any(np.abs(X_new) > 1e6) or np.any(np.std(X_new, axis=0) < 1e-10):
                return 1e6, {'mae': float('inf'), 'mse': float('inf')}

            mae_ensemble, mse_ensemble = self._evaluate_with_ensemble(X_new, y)
            
            if np.isnan(mae_ensemble) or np.isnan(mse_ensemble):
                return 1e6, {'mae': float('inf'), 'mse': float('inf')}

            complexity = sum(tree.size() for tree in trees)
            penalty = 0
            fitness = mse_ensemble + penalty

            return fitness, {'mae': mae_ensemble, 'mse': mse_ensemble}

        except:
            return 1e6, {'mae': float('inf'), 'mse': float('inf')}
    
    def _evaluate_with_ensemble(self, X, y):
        """Evalúa usando ensemble de Ridge + RandomForest con CV."""
        try:
            ridge_model = Ridge(alpha=1.0, random_state=42)
            ridge_cv = cross_validate(
                ridge_model, X, y,
                cv=self.cv_strategy_,
                scoring={'mae': 'neg_mean_absolute_error', 'mse': 'neg_mean_squared_error'},
                n_jobs=-1
            )
            
            rf_model = RandomForestRegressor(
                n_estimators=50, # ¿Valor óptimo?
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_cv = cross_validate(
                rf_model, X, y,
                cv=self.cv_strategy_,
                scoring={'mae': 'neg_mean_absolute_error', 'mse': 'neg_mean_squared_error'},
                n_jobs=-1
            )
            
            if (np.any(np.isnan(ridge_cv['test_mae'])) or np.any(np.isnan(ridge_cv['test_mse'])) or
                np.any(np.isnan(rf_cv['test_mae'])) or np.any(np.isnan(rf_cv['test_mse']))):
                return float('inf'), float('inf')
            
            ridge_mae = -ridge_cv['test_mae'].mean()
            ridge_mse = -ridge_cv['test_mse'].mean()
            rf_mae = -rf_cv['test_mae'].mean()
            rf_mse = -rf_cv['test_mse'].mean()
            
            ensemble_mae = (ridge_mae + rf_mae) / 2.0
            ensemble_mse = (ridge_mse + rf_mse) / 2.0
            
            return ensemble_mae, ensemble_mse
            
        except Exception as e:
            return float('inf'), float('inf')
    
    def _evaluate_feature_subset_cv(self, selection, X_full, y_full):
        """Evalúa subconjunto de features usando Cross-Validation con ensemble."""
        if np.sum(selection) == 0:
            return 1e6, {'mae': float('inf'), 'mse': float('inf')}

        try:
            X_selected = X_full[:, selection]
            mae_ensemble, mse_ensemble = self._evaluate_with_ensemble(X_selected, y_full)
            
            if np.isnan(mae_ensemble) or np.isnan(mse_ensemble):
                return 1e6, {'mae': float('inf'), 'mse': float('inf')}

            n_features = np.sum(selection)
            penalty = 0

            return mse_ensemble + penalty, {'mae': mae_ensemble, 'mse': mse_ensemble}
        except:
            return 1e6, {'mae': float('inf'), 'mse': float('inf')}
    
    def _transform_without_selection(self, X):
        """Transforma datos sin aplicar selección de features."""
        if not self.best_trees_:
            raise ValueError("No entrenado")
        
        if hasattr(X, 'values'):
            X = X.values
        
        X_scaled = self.scaler_.transform(X)
        X_new = X_scaled.copy()
        
        for tree in self.best_trees_:
            try:
                new_feature = tree.evaluate(X_scaled)
                new_feature = np.nan_to_num(new_feature, nan=0.0, posinf=100.0, neginf=-100.0)
                new_feature = np.clip(new_feature, -1000, 1000)
                X_new = np.column_stack([X_new, new_feature])
            except:
                X_new = np.column_stack([X_new, np.zeros(X_scaled.shape[0])])
        
        return X_new

    def transform(self, X):
        """Transforma datos usando los mejores árboles y aplica selección de features."""
        X_transformed = self._transform_without_selection(X)
        # Primeros transformamos y luego seleccionamos features
        if self.feature_selection_ is not None:
            X_transformed = X_transformed[:, self.feature_selection_]
        
        return X_transformed
    
    def _create_random_tree(self):
        """Crea conjunto aleatorio de árboles."""
        trees = []
        for _ in range(self.n_features_to_create):
            tree = self._generate_tree(max_depth=random.randint(2, self.max_depth))
            trees.append(tree)
        return trees
    
    def _generate_tree(self, max_depth):
        """Genera un árbol aleatorio."""
        if max_depth <= 1 or random.random() < 0.3:
            if random.random() < 0.8:
                return GPNode(random.randint(0, self.n_features_in_ - 1), node_type='terminal')
            else:
                return GPNode(random.uniform(-5, 5), node_type='terminal')
        else:
            func_name = random.choice(list(self.functions.keys()))
            arity = self.functions[func_name]
            children = [self._generate_tree(max_depth - 1) for _ in range(arity)]
            return GPNode(func_name, children, node_type='function')
    
    def _tournament_selection(self, population, fitness_scores):
        """Selección por torneo."""
        tournament_idx = random.sample(range(len(population)), 
                                     min(self.tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_idx]
        winner_idx = tournament_idx[np.argmin(tournament_fitness)]
        return [tree.copy() for tree in population[winner_idx]]
    
    def _crossover_trees(self, parent1, parent2):
        """Cruce entre conjuntos de árboles."""
        child1 = [tree.copy() for tree in parent1]
        child2 = [tree.copy() for tree in parent2]
        
        n_swap = random.randint(1, min(len(child1), len(child2)) // 2)
        indices = random.sample(range(min(len(child1), len(child2))), n_swap)
        
        for idx in indices:
            child1[idx], child2[idx] = child2[idx].copy(), child1[idx].copy()
        
        for i in range(min(len(child1), len(child2))):
            if random.random() < 0.3:
                child1[i], child2[i] = self._crossover_single_tree(child1[i], child2[i])
        
        return child1, child2
    
    def _crossover_single_tree(self, tree1, tree2):
        """Cruce entre dos árboles individuales."""
        if tree1.children and tree2.children and random.random() < 0.7:
            idx1 = random.randint(0, len(tree1.children) - 1)
            idx2 = random.randint(0, len(tree2.children) - 1)
            tree1.children[idx1], tree2.children[idx2] = tree2.children[idx2].copy(), tree1.children[idx1].copy()
        
        return tree1, tree2
    
    def _mutate_trees(self, trees):
        """Mutación de conjunto de árboles."""
        mutated = [tree.copy() for tree in trees]
        
        for i, tree in enumerate(mutated):
            if random.random() < 0.3:
                mutated[i] = self._mutate_single_tree(tree)
        
        if random.random() < 0.1:
            idx = random.randint(0, len(mutated) - 1)
            mutated[idx] = self._generate_tree(max_depth=random.randint(2, self.max_depth))
        
        return mutated
    
    def _mutate_single_tree(self, tree):
        """Mutación de un árbol individual."""
        mutated = tree.copy()
        
        if random.random() < 0.5 and mutated.children:
            mutated.value = random.choice(list(self.functions.keys()))
            required_arity = self.functions[mutated.value]
            while len(mutated.children) < required_arity:
                mutated.children.append(self._generate_tree(max_depth=2))
            mutated.children = mutated.children[:required_arity]
        
        elif mutated.node_type == 'terminal':
            if isinstance(mutated.value, int):
                mutated.value = random.randint(0, self.n_features_in_ - 1)
            else:
                mutated.value = random.uniform(-5, 5)
        
        for child in mutated.children:
            if random.random() < 0.2:
                child = self._mutate_single_tree(child)
        
        return mutated
    
    def _tournament_selection_fs(self, population, fitness_scores, tournament_size):
        """Selección por torneo para feature selection."""
        tournament_idx = random.sample(range(len(population)), 
                                     min(tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_idx]
        winner_idx = tournament_idx[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover_fs(self, parent1, parent2):
        """Cruce uniforme para feature selection."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        mask = np.random.rand(len(parent1)) < 0.5
        child1[mask] = parent2[mask]
        child2[mask] = parent1[mask]
        
        if np.sum(child1) < 2:
            child1[random.randint(0, len(child1)-1)] = True
            child1[random.randint(0, len(child1)-1)] = True
        if np.sum(child2) < 2:
            child2[random.randint(0, len(child2)-1)] = True
            child2[random.randint(0, len(child2)-1)] = True
        
        return child1, child2
    
    def _mutate_fs(self, individual):
        """Mutación para feature selection."""
        mutated = individual.copy()
        
        n_flips = random.randint(1, 3)
        for _ in range(n_flips):
            idx = random.randint(0, len(mutated) - 1)
            mutated[idx] = not mutated[idx]
        
        if np.sum(mutated) < 2:
            mutated[random.randint(0, len(mutated)-1)] = True
            mutated[random.randint(0, len(mutated)-1)] = True
        
        return mutated