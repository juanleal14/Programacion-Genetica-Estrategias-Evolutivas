import numpy as np
import pandas as pd
import time
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')


class GPNode:
    """Nodo de árbol de programación genética."""
    
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
                return np.clip(val, -100, 100) ** 2
            elif self.value == 'log':
                return np.log(np.abs(self.children[0].evaluate(X)) + 1)
            #elif self.value == 'abs':
            #    return np.abs(self.children[0].evaluate(X))
            elif self.value == 'sin':
                return np.sin(self.children[0].evaluate(X))
            elif self.value == 'cos':
                return np.cos(self.children[0].evaluate(X))
            elif self.value == 'tanh':
                return np.tanh(self.children[0].evaluate(X))
    
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
    """Optimizador de programación genética para crear features."""
    
    def __init__(self, maxtime=1200, population_size=100, n_features_to_create=4,
                 mutation_prob=0.2, crossover_prob=0.8, tournament_size=5,
                 max_depth=5, elite_size=0.1, apply_feature_selection=True,
                 evaluation_model='ridge', random_state=100473223):
        self.maxtime = maxtime
        self.population_size = population_size
        self.n_features_to_create = n_features_to_create
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        self.elite_size = int(population_size * elite_size)
        self.apply_feature_selection = apply_feature_selection
        self.evaluation_model = evaluation_model  # 'ridge' o 'linear'
        self.random_state = random_state
        
        # Funciones disponibles
        self.functions = {
            'add': 2, 'sub': 2, 'mul': 2, 'div': 2,
            'sqrt': 1, 'square': 1, 'log': 1, 'sin': 1, 'cos': 1, 'tanh': 1
        }
        
        self.best_trees_ = []
        self.best_fitness_ = float('inf')
        self.fitness_history_ = []
        self.best_metrics_ = {'mae': None, 'mse': None}
        self.feature_selection_ = None
    
    def fit(self, X, y):
        """Entrena usando programación genética y selección de features."""
        # Fijar semilla si se especifica
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)
        
        total_time = self.maxtime
        gp_time = total_time * 0.7  # 70% para GP
        fs_time = total_time * 0.3  # 30% para Feature Selection
        
        start_time = time.time()
        
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # División train/validation para early stopping
        n_val = int(len(X) * 0.2)
        indices = np.random.permutation(len(X))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        self.scaler_ = RobustScaler()
        X_train_scaled = self.scaler_.fit_transform(X_train)
        X_val_scaled = self.scaler_.transform(X_val)
        self.n_features_in_ = X.shape[1]
        
        print(f"\n{'='*70}")
        print(f"PROGRAMACIÓN GENÉTICA")
        print(f"{'='*70}")
        print(f"Población: {self.population_size} | Features a crear: {self.n_features_to_create}")
        print(f"Profundidad máxima: {self.max_depth}")
        print(f"Modelo de evaluación: {self.evaluation_model.upper()}")
        print(f"Tiempo asignado GP: {gp_time/60:.1f}min ({gp_time}s)")
        print(f"Tiempo asignado FS: {fs_time/60:.1f}min ({fs_time}s)")
        print(f"{'='*70}\n")
        
        # Inicializar población de árboles
        population = [self._create_random_tree() for _ in range(self.population_size)]
        
        generation = 0
        gp_early_stop = 0
        best_val_fitness = float('inf')
        gp_start = time.time()
        
        # FASE 1: PROGRAMACIÓN GENÉTICA (70% del tiempo)
        while (time.time() - gp_start) < gp_time:
            generation += 1
            
            # Evaluar población en train y validación
            train_fitness = []
            val_fitness = []
            for individual in population:
                train_fit, _ = self._evaluate_individual(individual, X_train_scaled, y_train)
                val_fit, val_metrics = self._evaluate_individual(individual, X_val_scaled, y_val)
                train_fitness.append(train_fit)
                val_fitness.append(val_fit)

            # Actualizar mejor basado en VALIDACIÓN
            best_val_idx = np.argmin(val_fitness)
            if val_fitness[best_val_idx] < best_val_fitness:
                best_val_fitness = val_fitness[best_val_idx]
                self.best_fitness_ = train_fitness[best_val_idx]
                self.best_trees_ = [tree.copy() for tree in population[best_val_idx]]
                self.best_metrics_ = val_metrics
                gp_early_stop = 0
                print(f"Gen {generation} - MEJORA! Val: {best_val_fitness:.4f} | Train: {self.best_fitness_:.4f}")
            else:
                gp_early_stop += 1
            
            # Early stopping para GP
            if gp_early_stop >= 100:  # 100 generaciones sin mejora en validación
                print(f"GP Early stopping en generación {generation}")
                break
            
            # CAMBIO DE RAMA cada 200 generaciones de estancamiento
            #if stagnation > 0 and stagnation % 200 == 0:
            #    print(f"\n🔄 CAMBIO DE RAMA en generación {generation}")
            #    print(f"   Estancamiento: {stagnation} generaciones")
            #    print(f"   Diversificando población...")
            #    
            #    elite_size_restart = max(1, self.population_size // 5)
            #    elite_indices = np.argsort(fitness_scores)[:elite_size_restart]
            #    
            #    new_population_restart = []
            #    
            #    for idx in elite_indices:
            #        new_population_restart.append([tree.copy() for tree in population[idx]])
            #    
            #    while len(new_population_restart) < self.population_size:
            #        new_individual = []
            #        for _ in range(self.n_features_to_create):
            #            depth = random.randint(3, self.max_depth + 1)
            #            tree = self._generate_tree(max_depth=depth)
            #            new_individual.append(tree)
            #        new_population_restart.append(new_individual)
            #    
            #    population = new_population_restart
            #    print(f"   Nueva población creada con {len(population)} individuos")
            
            self.fitness_history_.append(self.best_fitness_)
            
            # Nueva generación basada en validación
            new_population = []
            
            # Elitismo basado en validación
            elite_indices = np.argsort(val_fitness)[:self.elite_size]
            for idx in elite_indices:
                new_population.append([tree.copy() for tree in population[idx]])
            
            # Generar resto
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, val_fitness)
                parent2 = self._tournament_selection(population, val_fitness)
                
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
            
            population = new_population[:self.population_size]
            
            if generation % 50 == 0:
                elapsed = time.time() - gp_start
                print(f"Gen {generation} | Val: {best_val_fitness:.4f} | Train: {self.best_fitness_:.4f} | " +
                      f"Tiempo GP: {elapsed/60:.1f}min | Early stop: {gp_early_stop}")
        
        gp_elapsed = time.time() - gp_start
        print(f"\nProgramación Genética completada en {generation} generaciones ({gp_elapsed/60:.1f}min)")
        if (
            self.best_metrics_['mse'] is not None
            and self.best_metrics_['mae'] is not None
            and np.isfinite(self.best_metrics_['mse'])
            and np.isfinite(self.best_metrics_['mae'])
        ):
            print(f"Mejor MSE: {self.best_metrics_['mse']:.4f} | Mejor MAE: {self.best_metrics_['mae']:.4f}")
        print(f"Mejores árboles encontrados:")
        for i, tree in enumerate(self.best_trees_):
            print(f"  {i+1}: {tree.to_string()}")

        # FASE 2: FEATURE SELECTION EVOLUTIVA (30% del tiempo)
        if self.apply_feature_selection:
            print(f"\n{'='*70}")
            print(f"FEATURE SELECTION EVOLUTIVA")
            print(f"{'='*70}")
            
            # Transformar el dataset completo con las features generadas
            X_transformed = self._transform_without_selection(X)
            
            print(f"Aplicando selección evolutiva sobre {X_transformed.shape[1]} features...")
            print(f"Usando cross-validation (5 folds) para más robustez...")
            print(f"Tiempo máximo: {fs_time/60:.1f}min ({fs_time}s)")
            
            # Aplicar selección evolutiva usando CV con límite de tiempo
            fs_start = time.time()
            self.feature_selection_, fs_metrics = self._evolutionary_feature_selection_cv(
                X_transformed, y,
                population_size=30,
                max_time=fs_time  # Pasar límite de tiempo
            )
            fs_elapsed = time.time() - fs_start
            
            n_selected = np.sum(self.feature_selection_)
            print(f"\n✓ Selección completada: {n_selected}/{len(self.feature_selection_)} features seleccionadas")
            print(f"  Mejor MAE (CV): {fs_metrics['mae']:.4f}")
            print(f"  Mejor MSE (CV): {fs_metrics['mse']:.4f}")
            print(f"  Tiempo usado: {fs_elapsed/60:.1f}min")

        total_elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"ENTRENAMIENTO COMPLETADO")
        print(f"{'='*70}")
        print(f"Tiempo total: {total_elapsed/60:.1f}min")
        print(f"  - Programación Genética: {gp_elapsed/60:.1f}min ({gp_elapsed/total_elapsed*100:.1f}%)")
        if self.apply_feature_selection:
            print(f"  - Feature Selection: {fs_elapsed/60:.1f}min ({fs_elapsed/total_elapsed*100:.1f}%)")
        print(f"{'='*70}")

        return self
    
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
    
    def _get_evaluation_model(self):
        """Retorna el modelo a usar para evaluación."""
        if self.evaluation_model == 'ridge':
            # Ridge con regularización adaptativa
            return Ridge(alpha=1.0, random_state=42)
        else:
            return LinearRegression()
    def transform(self, X):
        """Transforma datos usando los mejores árboles y aplica selección de features."""
        X_transformed = self._transform_without_selection(X)
        
        # Aplicar selección de features si existe
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
    
    def _evaluate_individual(self, trees, X, y):
        """Evalúa un individuo (conjunto de árboles)."""
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

            # Usar el modelo configurado (Ridge o Linear) con 5-fold CV
            model = self._get_evaluation_model()
            cv_results = cross_validate(
                model,
                X_new,
                y,
                cv=3,  # 5-fold para más robustez
                scoring={'mae': 'neg_mean_absolute_error', 'mse': 'neg_mean_squared_error'},
                n_jobs=-1
            )

            if (np.any(np.isnan(cv_results['test_mae'])) or
                    np.any(np.isnan(cv_results['test_mse']))):
                return 1e6, {'mae': float('inf'), 'mse': float('inf')}

            mae = -cv_results['test_mae'].mean()
            mse = -cv_results['test_mse'].mean()

            complexity = sum(tree.size() for tree in trees)
            penalty = 0.001 * complexity
            fitness = mse + penalty

            return fitness, {'mae': mae, 'mse': mse}

        except:
            return 1e6, {'mae': float('inf'), 'mse': float('inf')}
    
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
    
    def _evolutionary_feature_selection_cv(self, X_full, y_full, 
                                           population_size=30, max_time=None):
        """Selección evolutiva de features usando Cross-Validation con límite de tiempo."""
        n_features = X_full.shape[1]
        
        # División train/validation para FS
        n_val = int(len(X_full) * 0.2)
        indices = np.random.permutation(len(X_full))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        
        X_fs_train, X_fs_val = X_full[train_idx], X_full[val_idx]
        y_fs_train, y_fs_val = y_full[train_idx], y_full[val_idx]
        
        # Inicializar población
        population = []
        for _ in range(population_size):
            individual = np.zeros(n_features, dtype=bool)
            n_selected = random.randint(3, min(15, n_features))
            selected_idx = random.sample(range(n_features), n_selected)
            individual[selected_idx] = True
            population.append(individual)
        
        best_val_fitness = float('inf')
        best_individual = None
        best_metrics = {'mae': None, 'mse': None}
        fs_early_stop = 0
        
        fs_start = time.time()
        gen = 0

        while True:
            gen += 1
            
            # Verificar límite de tiempo si existe
            if max_time is not None and (time.time() - fs_start) >= max_time:
                print(f"  Límite de tiempo alcanzado ({max_time}s)")
                break
            
            train_fitness = []
            val_fitness = []
            for individual in population:
                train_fit, _ = self._evaluate_feature_subset_simple(individual, X_fs_train, y_fs_train)
                val_fit, val_metrics = self._evaluate_feature_subset_simple(individual, X_fs_val, y_fs_val)
                train_fitness.append(train_fit)
                val_fitness.append(val_fit)

            current_best_idx = np.argmin(val_fitness)
            if val_fitness[current_best_idx] < best_val_fitness:
                best_val_fitness = val_fitness[current_best_idx]
                best_individual = population[current_best_idx].copy()
                best_metrics = val_metrics
                fs_early_stop = 0
                if gen % 20 == 0 or gen < 10:
                    n_selected = np.sum(best_individual)
                    elapsed = time.time() - fs_start
                    print(f"  Gen {gen}: Val MSE = {best_metrics['mse']:.4f} | Features: {n_selected} | Tiempo: {elapsed:.1f}s")
            else:
                fs_early_stop += 1
            
            # Early stopping para FS
            if fs_early_stop >= 30:  # 30 generaciones sin mejora
                print(f"  FS Early stopping en generación {gen}")
                break
            
            # Nueva generación basada en validación
            new_population = []
            
            elite_size = max(1, population_size // 10)
            elite_indices = np.argsort(val_fitness)[:elite_size]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            while len(new_population) < population_size:
                parent1 = self._tournament_selection_fs(population, val_fitness, 3)
                parent2 = self._tournament_selection_fs(population, val_fitness, 3)
                
                if random.random() < 0.8:
                    child1, child2 = self._crossover_fs(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                if random.random() < 0.3:
                    child1 = self._mutate_fs(child1)
                if random.random() < 0.3:
                    child2 = self._mutate_fs(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:population_size]
        
        print(f"  Selección completada en {gen} generaciones")
        return best_individual, best_metrics
    
    def _evaluate_feature_subset_simple(self, selection, X, y):
        """Evalúa subconjunto de features sin CV (más rápido)."""
        if np.sum(selection) == 0:
            return 1e6, {'mae': float('inf'), 'mse': float('inf')}

        try:
            X_selected = X[:, selection]
            model = self._get_evaluation_model()
            model.fit(X_selected, y)
            y_pred = model.predict(X_selected)
            
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            # Penalización por muchas features
            n_features = np.sum(selection)
            penalty = 0.01 * n_features

            return mse + penalty, {'mae': mae, 'mse': mse}
        except:
            return 1e6, {'mae': float('inf'), 'mse': float('inf')}
    
    def _evaluate_feature_subset_cv(self, selection, X_full, y_full, cv_folds=3):
        """Evalúa un subconjunto de features usando Cross-Validation."""
        if np.sum(selection) == 0:
            return 1e6, {'mae': float('inf'), 'mse': float('inf')}

        try:
            X_selected = X_full[:, selection]

            # Usar el modelo configurado (Ridge o Linear)
            model = self._get_evaluation_model()
            
            # Cross-validation con más folds para más robustez
            cv_results = cross_validate(
                model,
                X_selected,
                y_full,
                cv=cv_folds,
                scoring={'mae': 'neg_mean_absolute_error', 'mse': 'neg_mean_squared_error'},
                n_jobs=-1
            )
            
            if (np.any(np.isnan(cv_results['test_mae'])) or
                    np.any(np.isnan(cv_results['test_mse']))):
                return 1e6, {'mae': float('inf'), 'mse': float('inf')}

            mae = -cv_results['test_mae'].mean()
            mse = -cv_results['test_mse'].mean()

            # Penalización más fuerte por muchas features (reducir overfitting)
            n_features = np.sum(selection)
            penalty = 0.01 * n_features  # 10x más penalización

            return mse + penalty, {'mae': mae, 'mse': mse}
        except:
            return 1e6, {'mae': float('inf'), 'mse': float('inf')}
    
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


# ============================================================================
# MAIN - EJEMPLO DE USO Y EVALUACIÓN
# ============================================================================
#
#if __name__ == "__main__":
#    # Cargar dataset
#    df = pd.read_csv('california.csv')
#    
#    # CASO DIABETES
#    #X = df.drop('target', axis=1).values
#    #y = df['target'].values
#    
#    # CASO CALIFORNIA (descomentar si se usa otro dataset)
#    X = df.drop('MedHouseVal', axis=1).values
#    y = df['MedHouseVal'].values
#    
#    # Split train/test
#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.2, random_state=42
#    )
#    
#    print(f"\n{'='*70}")
#    print(f"EVALUACIÓN DEL SISTEMA")
#    print(f"{'='*70}")
#    print(f"Dataset: {X.shape[0]} instancias, {X.shape[1]} features")
#    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
#    print(f"{'='*70}\n")
#    
#    # ========================================================================
#    # BASELINE: Modelo sin optimización
#    # ========================================================================
#    print(f"\n{'='*70}")
#    print(f"BASELINE (Sin Optimización)")
#    print(f"{'='*70}")
#    
#    scaler = RobustScaler()
#    X_train_scaled = scaler.fit_transform(X_train)
#    X_test_scaled = scaler.transform(X_test)
#    
#    baseline = Ridge(alpha=1.0, random_state=42)
#    baseline.fit(X_train, y_train)
#    baseline_preds = baseline.predict(X_test)
#    
#    baseline_mae = mean_absolute_error(y_test, baseline_preds)
#    baseline_mse = mean_squared_error(y_test, baseline_preds)
#    
#    print(f"MAE: {baseline_mae:.4f}")
#    print(f"MSE: {baseline_mse:.4f}")
#    print(f"Features utilizadas: {X_train.shape[1]}")
#    
#    # ========================================================================
#    # OPTIMIZACIÓN CON PROGRAMACIÓN GENÉTICA + FEATURE SELECTION
#    # ========================================================================
#    print(f"\n{'='*70}")
#    print(f"OPTIMIZACIÓN EVOLUTIVA")
#    print(f"{'='*70}")
#    
#    # Crear optimizador
#    gp_optimizer = EvolutionaryOptimizer(
#        maxtime=1200,  # 20 minutos (ajusta según necesites)
#    )
#    
#    # Entrenar el optimizador (aprende transformaciones)
#    gp_optimizer.fit(X_train, y_train)
#    
#    # Transformar los datos (aplicar las transformaciones aprendidas)
#    X_train_optimized = gp_optimizer.transform(X_train)
#    X_test_optimized = gp_optimizer.transform(X_test)
#    
#    # Entrenar UN NUEVO MODELO con los datos optimizados (simula lo que hace el profesor)
#    optimized_model = Ridge(alpha=1.0, random_state=42)
#    optimized_model.fit(X_train_optimized, y_train)
#    optimized_preds = optimized_model.predict(X_test_optimized)
#    
#    optimized_mae = mean_absolute_error(y_test, optimized_preds)
#    optimized_mse = mean_squared_error(y_test, optimized_preds)
#    
#    # ========================================================================
#    # RESULTADOS FINALES
#    # ========================================================================
#    print(f"\n{'='*70}")
#    print(f"RESULTADOS FINALES")
#    print(f"{'='*70}")
#    
#    print(f"\nBaseline (sin optimización):")
#    print(f"  MAE: {baseline_mae:.4f}")
#    print(f"  MSE: {baseline_mse:.4f}")
#    print(f"  Features: {X_train.shape[1]}")
#    
#    print(f"\nCon Optimización Evolutiva:")
#    print(f"  MAE: {optimized_mae:.4f}")
#    print(f"  MSE: {optimized_mse:.4f}")
#    print(f"  Features: {X_train_optimized.shape[1]}")
#    
#    # Calcular mejoras
#    mae_improvement = ((baseline_mae - optimized_mae) / baseline_mae * 100)
#    mse_improvement = ((baseline_mse - optimized_mse) / baseline_mse * 100)
#    
#    print(f"\n{'='*70}")
#    print(f"MEJORAS OBTENIDAS")
#    print(f"{'='*70}")
#    print(f"Mejora en MAE: {mae_improvement:+.2f}%")
#    print(f"Mejora en MSE: {mse_improvement:+.2f}%")
#    
#    if gp_optimizer.feature_selection_ is not None:
#        n_selected = np.sum(gp_optimizer.feature_selection_)
#        n_total = len(gp_optimizer.feature_selection_)
#        print(f"Features seleccionadas: {n_selected}/{n_total}")
#        
#        # Mostrar cuáles features se seleccionaron
#        print(f"\nFeatures seleccionadas:")
#        selected_indices = np.where(gp_optimizer.feature_selection_)[0]
#        for idx in selected_indices:
#            if idx < X.shape[1]:
#                print(f"  X{idx} (original)")
#            else:
#                tree_idx = idx - X.shape[1]
#                if tree_idx < len(gp_optimizer.best_trees_):
#                    print(f"  {gp_optimizer.best_trees_[tree_idx].to_string()} (generada)")
#    
#    print(f"\n{'='*70}")
#    print(f"¡Evaluación completada!")
#    print(f"{'='*70}\n")