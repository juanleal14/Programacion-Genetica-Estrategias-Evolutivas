import numpy as np
import pandas as pd
import time
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')


class GPNode:
    """Nodo de árbol de programación genética."""
    
    def __init__(self, value, children=None, node_type='terminal'):
        self.value = value
        self.children = children or []
        self.node_type = node_type  # 'terminal' o 'function'
    
    def evaluate(self, X):
        """Evalúa el nodo con los datos X."""
        if self.node_type == 'terminal':
            if isinstance(self.value, int):  # Índice de feature
                return X[:, self.value]
            else:  # Constante
                return np.full(X.shape[0], self.value)
        
        else:  # function
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
            elif self.value == 'abs':
                return np.abs(self.children[0].evaluate(X))
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
    
    def __init__(self, maxtime=1200, population_size=50, n_features_to_create=8,
                 mutation_prob=0.2, crossover_prob=0.8, tournament_size=5,
                 max_depth=5, elite_size=0.1):
        self.maxtime = maxtime
        self.population_size = population_size
        self.n_features_to_create = n_features_to_create
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        self.elite_size = int(population_size * elite_size)
        
        # Funciones disponibles
        self.functions = {
            'add': 2, 'sub': 2, 'mul': 2, 'div': 2,  # Binarias
            'sqrt': 1, 'square': 1, 'log': 1, 'abs': 1, 'sin': 1, 'cos': 1, 'tanh': 1  # Unarias
        }
        
        self.best_trees_ = []
        self.best_fitness_ = float('inf')
        self.fitness_history_ = []
    
    def fit(self, X, y):
        """Entrena usando programación genética."""
        start_time = time.time()
        
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.n_features_in_ = X.shape[1]
        
        print(f"\n{'='*70}")
        print(f"PROGRAMACIÓN GENÉTICA")
        print(f"{'='*70}")
        print(f"Población: {self.population_size} | Features a crear: {self.n_features_to_create}")
        print(f"Profundidad máxima: {self.max_depth} | Tiempo: {self.maxtime/60:.1f}min")
        print(f"{'='*70}\n")
        
        # Inicializar población de árboles
        population = [self._create_random_tree() for _ in range(self.population_size)]
        
        generation = 0
        stagnation = 0
        
        while (time.time() - start_time) < self.maxtime:
            generation += 1
            
            # Evaluar población
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_individual(individual, X_scaled, y)
                fitness_scores.append(fitness)
            
            # Actualizar mejor
            best_idx = np.argmin(fitness_scores)
            if fitness_scores[best_idx] < self.best_fitness_:
                self.best_fitness_ = fitness_scores[best_idx]
                self.best_trees_ = population[best_idx].copy()
                stagnation = 0
                print(f"Gen {generation} - MEJORA! Fitness: {self.best_fitness_:.4f}")
                print(f"  Mejor árbol: {self.best_trees_[0].to_string()}")
            else:
                stagnation += 1
            
            # CAMBIO DE RAMA cada 200 generaciones de estancamiento
            if stagnation > 0 and stagnation % 200 == 0:
                print(f"\n🔄 CAMBIO DE RAMA en generación {generation}")
                print(f"   Estancamiento: {stagnation} generaciones")
                print(f"   Diversificando población...")
                
                # Mantener solo el 20% de elite
                elite_size_restart = max(1, self.population_size // 5)
                elite_indices = np.argsort(fitness_scores)[:elite_size_restart]
                
                # Crear nueva población diversa
                new_population_restart = []
                
                # Mantener elite
                for idx in elite_indices:
                    new_population_restart.append([tree.copy() for tree in population[idx]])
                
                # Generar resto completamente nuevo con más diversidad
                while len(new_population_restart) < self.population_size:
                    # Crear individuos más diversos
                    new_individual = []
                    for _ in range(self.n_features_to_create):
                        # Árboles más profundos y diversos
                        depth = random.randint(3, self.max_depth + 1)
                        tree = self._generate_tree(max_depth=depth)
                        new_individual.append(tree)
                    new_population_restart.append(new_individual)
                
                population = new_population_restart
                print(f"   Nueva población creada con {len(population)} individuos")
            
            self.fitness_history_.append(self.best_fitness_)
            
            # Nueva generación
            new_population = []
            
            # Elitismo
            elite_indices = np.argsort(fitness_scores)[:self.elite_size]
            for idx in elite_indices:
                new_population.append([tree.copy() for tree in population[idx]])
            
            # Generar resto
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
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
                elapsed = time.time() - start_time
                print(f"Gen {generation} | Fitness: {self.best_fitness_:.4f} | " +
                      f"Tiempo: {elapsed/60:.1f}min | Estancamiento: {stagnation}")
        
        print(f"\nCompletado en {generation} generaciones")
        print(f"Mejores árboles encontrados:")
        for i, tree in enumerate(self.best_trees_):
            print(f"  {i+1}: {tree.to_string()}")
        
        return self
    
    def transform(self, X):
        """Transforma datos usando los mejores árboles."""
        if not self.best_trees_:
            raise ValueError("No entrenado")
        
        if hasattr(X, 'values'):
            X = X.values
        
        X_scaled = self.scaler_.transform(X)
        X_new = X_scaled.copy()
        
        # Aplicar cada árbol como nueva feature
        for tree in self.best_trees_:
            try:
                new_feature = tree.evaluate(X_scaled)
                # Limpieza numérica
                new_feature = np.nan_to_num(new_feature, nan=0.0, posinf=100.0, neginf=-100.0)
                new_feature = np.clip(new_feature, -1000, 1000)
                X_new = np.column_stack([X_new, new_feature])
            except:
                X_new = np.column_stack([X_new, np.zeros(X_scaled.shape[0])])
        
        return X_new
    
    def predict(self, X):
        """Predice usando regresión lineal sobre features transformadas."""
        X_transformed = self.transform(X)
        return self.model_.predict(X_transformed)
    
    def _create_random_tree(self):
        """Crea conjunto aleatorio de árboles."""
        trees = []
        for _ in range(self.n_features_to_create):
            tree = self._generate_tree(max_depth=random.randint(2, self.max_depth))
            trees.append(tree)
        return trees
    
    def _generate_tree(self, max_depth):
        """Genera un árbol aleatorio."""
        if max_depth <= 1 or random.random() < 0.3:  # Terminal
            if random.random() < 0.8:  # Feature
                return GPNode(random.randint(0, self.n_features_in_ - 1), node_type='terminal')
            else:  # Constante
                return GPNode(random.uniform(-5, 5), node_type='terminal')
        
        else:  # Función
            func_name = random.choice(list(self.functions.keys()))
            arity = self.functions[func_name]
            children = [self._generate_tree(max_depth - 1) for _ in range(arity)]
            return GPNode(func_name, children, node_type='function')
    
    def _evaluate_individual(self, trees, X, y):
        """Evalúa un individuo (conjunto de árboles)."""
        try:
            # Crear features usando los árboles
            X_new = X.copy()
            for tree in trees:
                try:
                    new_feature = tree.evaluate(X)
                    new_feature = np.nan_to_num(new_feature, nan=0.0, posinf=100.0, neginf=-100.0)
                    new_feature = np.clip(new_feature, -1000, 1000)
                    X_new = np.column_stack([X_new, new_feature])
                except:
                    X_new = np.column_stack([X_new, np.zeros(X.shape[0])])
            
            # Verificar estabilidad
            if np.any(np.abs(X_new) > 1e6) or np.any(np.std(X_new, axis=0) < 1e-10):
                return 1e6
            
            # Cross-validation
            model = LinearRegression()
            cv_scores = cross_val_score(model, X_new, y, cv=3, 
                                      scoring='neg_mean_absolute_error', n_jobs=-1)
            
            if np.any(np.isnan(cv_scores)):
                return 1e6
            
            mae = -cv_scores.mean()
            
            # Penalización por complejidad
            complexity = sum(tree.size() for tree in trees)
            penalty = 0.001 * complexity
            
            # Guardar modelo si es el mejor hasta ahora
            if mae + penalty < self.best_fitness_:
                model.fit(X_new, y)
                self.model_ = model
            
            return mae + penalty
            
        except:
            return 1e6
    
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
        
        # Intercambiar algunos árboles
        n_swap = random.randint(1, min(len(child1), len(child2)) // 2)
        indices = random.sample(range(min(len(child1), len(child2))), n_swap)
        
        for idx in indices:
            child1[idx], child2[idx] = child2[idx].copy(), child1[idx].copy()
        
        # Cruce dentro de árboles individuales
        for i in range(min(len(child1), len(child2))):
            if random.random() < 0.3:
                child1[i], child2[i] = self._crossover_single_tree(child1[i], child2[i])
        
        return child1, child2
    
    def _crossover_single_tree(self, tree1, tree2):
        """Cruce entre dos árboles individuales."""
        # Seleccionar nodos aleatorios para intercambiar
        if tree1.children and tree2.children and random.random() < 0.7:
            # Intercambiar subárboles
            idx1 = random.randint(0, len(tree1.children) - 1)
            idx2 = random.randint(0, len(tree2.children) - 1)
            tree1.children[idx1], tree2.children[idx2] = tree2.children[idx2].copy(), tree1.children[idx1].copy()
        
        return tree1, tree2
    
    def _mutate_trees(self, trees):
        """Mutación de conjunto de árboles."""
        mutated = [tree.copy() for tree in trees]
        
        for i, tree in enumerate(mutated):
            if random.random() < 0.3:  # Mutar árbol
                mutated[i] = self._mutate_single_tree(tree)
        
        # Ocasionalmente reemplazar árbol completo
        if random.random() < 0.1:
            idx = random.randint(0, len(mutated) - 1)
            mutated[idx] = self._generate_tree(max_depth=random.randint(2, self.max_depth))
        
        return mutated
    
    def _mutate_single_tree(self, tree):
        """Mutación de un árbol individual."""
        mutated = tree.copy()
        
        if random.random() < 0.5 and mutated.children:  # Mutar función
            mutated.value = random.choice(list(self.functions.keys()))
            # Ajustar hijos si es necesario
            required_arity = self.functions[mutated.value]
            while len(mutated.children) < required_arity:
                mutated.children.append(self._generate_tree(max_depth=2))
            mutated.children = mutated.children[:required_arity]
        
        elif mutated.node_type == 'terminal':  # Mutar terminal
            if isinstance(mutated.value, int):  # Feature
                mutated.value = random.randint(0, self.n_features_in_ - 1)
            else:  # Constante
                mutated.value = random.uniform(-5, 5)
        
        # Mutar recursivamente hijos
        for child in mutated.children:
            if random.random() < 0.2:
                child = self._mutate_single_tree(child)
        
        return mutated


# Ejemplo de uso
if __name__ == "__main__":
    df = pd.read_csv('diabetes.csv')
    # CASO DIABETES
    X = df.drop('target', axis=1).values
    y = df['target'].values
    # CASO CALIFORNIA
    #X = df.drop('MedHouseVal', axis=1).values
    #y = df['MedHouseVal'].values
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Baseline
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    baseline = LinearRegression()
    baseline.fit(X_train_scaled, y_train)
    baseline_mae = mean_absolute_error(y_test, baseline.predict(X_test_scaled))
    #baseline_r2 = r2_score(y_test, baseline.predict(X_test_scaled))
    baseline_mse = np.mean((y_test - baseline.predict(X_test_scaled)) ** 2)
    print(f"\n{'='*60}")
    print(f"Baseline - MAE: {baseline_mae:.4f}")#, R²: {baseline_r2:.4f}")
    print(f"Baseline - MSE: {baseline_mse:.4f}")

    # Programación genética
    gp_optimizer = EvolutionaryOptimizer(
        maxtime=1200  # 20 minutos
    )
    
    gp_optimizer.fit(X_train, y_train)
    y_pred = gp_optimizer.predict(X_test)
    
    gp_mae = mean_absolute_error(y_test, y_pred)
    #gp_r2 = r2_score(y_test, y_pred)
    gp_mse = np.mean((y_test - y_pred) ** 2)
    print(f"\nResultados:")
    print(f"Baseline: MAE={baseline_mae:.4f}")
    print(f"GP:       MAE={gp_mae:.4f}")
    print(f"Mejora:   {((baseline_mae - gp_mae) / baseline_mae * 100):+.2f}%")
    print(f"Baseline: MSE={baseline_mse:.4f}")
    print(f"GP:       MSE={gp_mse:.4f}")
    print(f"Mejora:   {((baseline_mse - gp_mse) / baseline_mse * 100):+.2f}%")

    # EVALUAR CADA ÁRBOL INDIVIDUALMENTE
    print(f"\n{'='*60}")
    print(f"EVALUACIÓN INDIVIDUAL DE LOS {len(gp_optimizer.best_trees_)} ÁRBOLES")
    print(f"{'='*60}")
    
    for i, tree in enumerate(gp_optimizer.best_trees_, 1):
        try:
            # Crear dataset con solo este árbol
            tree_feature_train = tree.evaluate(X_train_scaled)
            tree_feature_test = tree.evaluate(X_test_scaled)
            
            # Limpieza
            tree_feature_train = np.nan_to_num(tree_feature_train, nan=0.0, posinf=100.0, neginf=-100.0)
            tree_feature_test = np.nan_to_num(tree_feature_test, nan=0.0, posinf=100.0, neginf=-100.0)
            tree_feature_train = np.clip(tree_feature_train, -1000, 1000)
            tree_feature_test = np.clip(tree_feature_test, -1000, 1000)
            
            # Combinar con features originales
            X_train_with_tree = np.column_stack([X_train_scaled, tree_feature_train])
            X_test_with_tree = np.column_stack([X_test_scaled, tree_feature_test])
            
            # Entrenar modelo
            tree_model = LinearRegression()
            tree_model.fit(X_train_with_tree, y_train)
            tree_pred = tree_model.predict(X_test_with_tree)
            
            tree_mae = mean_absolute_error(y_test, tree_pred)
            tree_improvement = ((baseline_mae - tree_mae) / baseline_mae * 100)
            
            print(f"Árbol {i}: {tree.to_string()}")
            print(f"  MAE: {tree_mae:.4f} | Mejora: {tree_improvement:+.2f}%")
            print(f"  Tamaño: {tree.size()} nodos | Profundidad: {tree.depth()}")
            print()
            
        except Exception as e:
            print(f"Árbol {i}: {tree.to_string()}")
            print(f"  ERROR: {str(e)}")
            print()