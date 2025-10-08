import numpy as np
import pandas as pd
import time
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
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
        """Evalúa el nodo con los datos X (array 2D escalado)."""
        if self.node_type == 'terminal':
            if isinstance(self.value, int):  # Índice de feature
                return X[:, self.value]
            else:  # Constante
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
            elif self.value == 'abs':
                return np.abs(self.children[0].evaluate(X))
            elif self.value == 'sin':
                return np.sin(self.children[0].evaluate(X))
            elif self.value == 'cos':
                return np.cos(self.children[0].evaluate(X))
            elif self.value == 'tanh':
                return np.tanh(self.children[0].evaluate(X))

    def copy(self):
        return GPNode(self.value, [c.copy() for c in self.children], self.node_type)

    def size(self):
        return 1 + sum(child.size() for child in self.children)

    def depth(self):
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def to_string(self):
        if self.node_type == 'terminal':
            return f"X{self.value}" if isinstance(self.value, int) else f"{self.value:.3f}"
        else:
            if len(self.children) == 1:
                return f"{self.value}({self.children[0].to_string()})"
            else:
                return f"({self.children[0].to_string()} {self.value} {self.children[1].to_string()})"


class EvolutionaryOptimizer(BaseEstimator, TransformerMixin):
    """
    Fase 1: Programación Genética (GP) para sintetizar nuevas features.
    Fase 2: Algoritmo Genético (GA) para seleccionar un subconjunto de features (originales + sintetizadas).
    """
    def __init__(
        self,
        # --- GP params ---
        maxtime=1200,
        population_size=50,
        n_features_to_create=8,
        mutation_prob=0.2,
        crossover_prob=0.8,
        tournament_size=5,
        max_depth=5,
        elite_frac=0.1,
        complexity_penalty=1e-3,
        # --- GA params ---
        do_feature_selection=True,
        fs_time_frac=0.3,                 # % del tiempo total para GA
        fs_population_size=60,
        fs_tournament_size=4,
        fs_crossover_prob=0.8,
        fs_mutation_prob=0.08,
        fs_elite_frac=0.1,
        fs_l1_penalty=1e-4,               # penaliza nº de columnas seleccionadas
        fs_cv=3
    ):
        # GP
        self.maxtime = maxtime
        self.population_size = population_size
        self.n_features_to_create = n_features_to_create
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        self.elite_size = int(population_size * elite_frac)
        self.complexity_penalty = complexity_penalty

        self.functions = {
            'add': 2, 'sub': 2, 'mul': 2, 'div': 2,
            'sqrt': 1, 'square': 1, 'log': 1, 'abs': 1, 'sin': 1, 'cos': 1, 'tanh': 1
        }

        self.best_trees_ = []
        self.best_fitness_ = float('inf')
        self.fitness_history_ = []

        # GA (feature selection)
        self.do_feature_selection = do_feature_selection
        self.fs_time_frac = fs_time_frac
        self.fs_population_size = fs_population_size
        self.fs_tournament_size = fs_tournament_size
        self.fs_crossover_prob = fs_crossover_prob
        self.fs_mutation_prob = fs_mutation_prob
        self.fs_elite_size = max(1, int(fs_population_size * fs_elite_frac))
        self.fs_l1_penalty = fs_l1_penalty
        self.fs_cv = fs_cv

        # Salidas de FS
        self.selected_mask_ = None  # máscara binaria sobre [X|Z]

    # ============================
    # API pública
    # ============================
    def fit(self, X, y):
        start_time = time.time()

        # Asegurar arrays
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        # Escalado robusto
        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.n_features_in_ = X.shape[1]

        # Partición de tiempo
        gp_budget = self.maxtime * (1.0 - (self.fs_time_frac if self.do_feature_selection else 0.0))
        ga_budget = self.maxtime - gp_budget

        print("\n" + "="*70)
        print("FASE 1 — PROGRAMACIÓN GENÉTICA (síntesis de atributos)")
        print("="*70)
        print(f"Población: {self.population_size} | Nuevas features por individuo: {self.n_features_to_create}")
        print(f"Profundidad máxima: {self.max_depth} | Tiempo GP: {gp_budget/60:.1f} min")
        print("="*70 + "\n")

        # ---------- Evolución GP ----------
        population = [self._create_random_individual() for _ in range(self.population_size)]
        generation = 0
        while (time.time() - start_time) < gp_budget:
            generation += 1
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_individual_gp(individual, X_scaled, y)
                fitness_scores.append(fitness)

            # Mejor individuo
            best_idx = int(np.argmin(fitness_scores))
            if fitness_scores[best_idx] < self.best_fitness_:
                self.best_fitness_ = float(fitness_scores[best_idx])
                self.best_trees_ = [t.copy() for t in population[best_idx]]
                print(f"Gen {generation} - MEJORA! Fitness: {self.best_fitness_:.6f}")
                if self.best_trees_:
                    print(f"  Árbol ejemplo: {self.best_trees_[0].to_string()}")
            self.fitness_history_.append(self.best_fitness_)

            # Nueva generación (elitismo + reproducción)
            new_population = []
            elite_indices = np.argsort(fitness_scores)[:self.elite_size]
            for idx in elite_indices:
                new_population.append([t.copy() for t in population[idx]])

            while len(new_population) < self.population_size:
                p1 = self._tournament_selection_gp(population, fitness_scores)
                p2 = self._tournament_selection_gp(population, fitness_scores)
                if random.random() < self.crossover_prob:
                    c1, c2 = self._crossover_individuals(p1, p2)
                else:
                    c1, c2 = [t.copy() for t in p1], [t.copy() for t in p2]
                if random.random() < self.mutation_prob:
                    c1 = self._mutate_individual(c1)
                if random.random() < self.mutation_prob:
                    c2 = self._mutate_individual(c2)
                new_population.extend([c1, c2])
            population = new_population[:self.population_size]

        # Construir X_optimized (train) con mejores árboles
        X_train_aug = self._augment_with_trees(X_scaled, self.best_trees_)

        # Modelo provisional con todo (por si no hay GA)
        self.model_ = LinearRegression().fit(X_train_aug, y)
        self.selected_mask_ = np.ones(X_train_aug.shape[1], dtype=bool)  # por defecto: todas

        # ---------- GA de selección de features ----------
        if self.do_feature_selection and (time.time() - start_time) < self.maxtime:
            remaining = self.maxtime - (time.time() - start_time)
            ga_budget = min(ga_budget, remaining)

            print("\n" + "="*70)
            print("FASE 2 — GA de Selección de Features (reducción de dimensionalidad)")
            print("="*70)
            print(f"Población GA: {self.fs_population_size} | Tiempo GA: {ga_budget/60:.1f} min")
            print(f"Penalización L1 (nº columnas): {self.fs_l1_penalty}")
            print("="*70 + "\n")

            mask, model = self._run_feature_selection_ga(
                X_train_aug, y, time_budget=ga_budget
            )
            if mask is not None and model is not None:
                self.selected_mask_ = mask
                self.model_ = model
                print(f"Selección GA: {self.selected_mask_.sum()}/{len(self.selected_mask_)} columnas usadas.")

        print("\nEntrenamiento completado.")
        return self

    def transform(self, X):
        if not isinstance(self.best_trees_, list) or len(self.best_trees_) == 0:
            raise ValueError("El optimizador no está entrenado (best_trees_ vacío).")
        if hasattr(X, 'values'):
            X = X.values
        X_scaled = self.scaler_.transform(X)
        X_aug = self._augment_with_trees(X_scaled, self.best_trees_)
        if self.selected_mask_ is None:
            return X_aug
        return X_aug[:, self.selected_mask_]

    def predict(self, X):
        X_transformed = self.transform(X)
        return self.model_.predict(X_transformed)

    # ============================
    # Helpers GP
    # ============================
    def _create_random_individual(self):
        return [self._generate_tree(max_depth=random.randint(2, self.max_depth))
                for _ in range(self.n_features_to_create)]

    def _generate_tree(self, max_depth):
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

    def _augment_with_trees(self, X_scaled, trees):
        X_new = X_scaled.copy()
        for tree in trees:
            try:
                feat = tree.evaluate(X_scaled)
                feat = np.nan_to_num(feat, nan=0.0, posinf=100.0, neginf=-100.0)
                feat = np.clip(feat, -1000, 1000)
                X_new = np.column_stack([X_new, feat])
            except Exception:
                X_new = np.column_stack([X_new, np.zeros(X_scaled.shape[0])])
        return X_new

    def _evaluate_individual_gp(self, trees, X, y):
        try:
            X_new = self._augment_with_trees(X, trees)
            # estabilidad
            if np.any(np.abs(X_new) > 1e6) or np.any(np.std(X_new, axis=0) < 1e-10):
                return 1e6
            model = LinearRegression()
            cv_scores = cross_val_score(model, X_new, y, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
            if np.any(np.isnan(cv_scores)):
                return 1e6
            mae = -cv_scores.mean()
            complexity = sum(t.size() for t in trees)
            penalty = self.complexity_penalty * complexity
            # Guardar modelo provisional si mejora
            if mae + penalty < self.best_fitness_:
                model.fit(X_new, y)
                self.model_ = model
            return mae + penalty
        except Exception:
            return 1e6

    def _tournament_selection_gp(self, population, fitness_scores):
        idxs = random.sample(range(len(population)), min(self.tournament_size, len(population)))
        fit = [fitness_scores[i] for i in idxs]
        win = idxs[int(np.argmin(fit))]
        return [t.copy() for t in population[win]]

    def _crossover_individuals(self, p1, p2):
        c1 = [t.copy() for t in p1]
        c2 = [t.copy() for t in p2]
        # Intercambio de árboles
        n_swap = random.randint(1, max(1, min(len(c1), len(c2)) // 2))
        indices = random.sample(range(min(len(c1), len(c2))), n_swap)
        for idx in indices:
            c1[idx], c2[idx] = c2[idx].copy(), c1[idx].copy()
        # Cruce dentro de algunos árboles
        for i in range(min(len(c1), len(c2))):
            if random.random() < 0.3:
                c1[i], c2[i] = self._crossover_single_tree(c1[i], c2[i])
        return c1, c2

    def _crossover_single_tree(self, t1, t2):
        if t1.children and t2.children and random.random() < 0.7:
            i1 = random.randint(0, len(t1.children) - 1)
            i2 = random.randint(0, len(t2.children) - 1)
            t1.children[i1], t2.children[i2] = t2.children[i2].copy(), t1.children[i1].copy()
        return t1, t2

    def _mutate_individual(self, trees):
        mutated = [t.copy() for t in trees]
        for i, tree in enumerate(mutated):
            if random.random() < 0.3:
                mutated[i] = self._mutate_single_tree(tree)
        if random.random() < 0.1:
            idx = random.randint(0, len(mutated) - 1)
            mutated[idx] = self._generate_tree(max_depth=random.randint(2, self.max_depth))
        return mutated

    def _mutate_single_tree(self, tree):
        m = tree.copy()
        if random.random() < 0.5 and m.children:
            m.value = random.choice(list(self.functions.keys()))
            req = self.functions[m.value]
            while len(m.children) < req:
                m.children.append(self._generate_tree(max_depth=2))
            m.children = m.children[:req]
        elif m.node_type == 'terminal':
            if isinstance(m.value, int):
                m.value = random.randint(0, self.n_features_in_ - 1)
            else:
                m.value = random.uniform(-5, 5)
        for i, child in enumerate(m.children):
            if random.random() < 0.2:
                m.children[i] = self._mutate_single_tree(child)
        return m

    # ============================
    # GA de selección de features
    # ============================
    def _run_feature_selection_ga(self, X_aug, y, time_budget):
        """
        GA sobre máscara binaria de longitud d = X_aug.shape[1].
        Fitness = MAE_CV + fs_l1_penalty * (#features activas).
        """
        start = time.time()
        d = X_aug.shape[1]
        if d == 0:
            return None, None

        # Población inicial: sesgo a mantener originales
        #   - asegurar que no haya máscaras "todo cero"
        pop = []
        for _ in range(self.fs_population_size):
            mask = np.zeros(d, dtype=bool)
            # Mantener de base ~70% originales y ~40% sintetizadas
            n_orig = self.n_features_in_
            base_orig = np.random.rand(n_orig) < 0.7
            base_synth = np.random.rand(d - n_orig) < 0.4
            mask[:n_orig] = base_orig
            mask[n_orig:] = base_synth
            if mask.sum() == 0:
                mask[np.random.randint(0, d)] = True
            pop.append(mask)

        best_mask = None
        best_fit = np.inf
        best_model = None

        gen = 0
        while (time.time() - start) < time_budget:
            gen += 1
            fits = []
            models = []

            for mask in pop:
                fit, model = self._fs_fitness(X_aug, y, mask)
                fits.append(fit)
                models.append(model)
                if fit < best_fit:
                    best_fit = fit
                    best_mask = mask.copy()
                    best_model = model

            # Reproducción GA
            new_pop = []
            elite_idx = np.argsort(fits)[:self.fs_elite_size]
            for idx in elite_idx:
                new_pop.append(pop[idx].copy())

            while len(new_pop) < self.fs_population_size:
                p1 = self._fs_tournament(pop, fits, self.fs_tournament_size)
                p2 = self._fs_tournament(pop, fits, self.fs_tournament_size)
                c1, c2 = self._fs_crossover(p1, p2, self.fs_crossover_prob)
                c1 = self._fs_mutate(c1, self.fs_mutation_prob)
                c2 = self._fs_mutate(c2, self.fs_mutation_prob)
                # Evitar todo ceros
                if not c1.any():
                    c1[np.random.randint(0, d)] = True
                if not c2.any():
                    c2[np.random.randint(0, d)] = True
                new_pop.extend([c1, c2])

            pop = new_pop[:self.fs_population_size]

            if gen % 20 == 0:
                print(f"  [GA] Gen {gen} | Mejor fitness: {best_fit:.6f} | Activas: {best_mask.sum()}")

        return best_mask, best_model

    def _fs_fitness(self, X_aug, y, mask):
        # Si máscara vacía: castigar
        if not mask.any():
            return 1e9, None
        X_sub = X_aug[:, mask]
        # Estabilidad
        if np.any(np.std(X_sub, axis=0) < 1e-12):
            return 1e8, None
        model = LinearRegression()
        scores = cross_val_score(model, X_sub, y, cv=self.fs_cv, scoring='neg_mean_absolute_error', n_jobs=-1)
        if np.any(np.isnan(scores)):
            return 1e8, None
        mae = -scores.mean()
        l1 = self.fs_l1_penalty * mask.sum()
        fitness = mae + l1
        # Guardar modelo entrenado (para predicción final)
        model.fit(X_sub, y)
        return fitness, model

    def _fs_tournament(self, pop, fits, k):
        idxs = random.sample(range(len(pop)), min(k, len(pop)))
        best = idxs[0]
        for i in idxs[1:]:
            if fits[i] < fits[best]:
                best = i
        return pop[best].copy()

    def _fs_crossover(self, m1, m2, p):
        if random.random() >= p or len(m1) != len(m2):
            return m1.copy(), m2.copy()
        d = len(m1)
        pt = random.randint(1, d - 1)
        c1 = np.concatenate([m1[:pt], m2[pt:]])
        c2 = np.concatenate([m2[:pt], m1[pt:]])
        return c1, c2

    def _fs_mutate(self, m, p):
        m = m.copy()
        flip = np.random.rand(len(m)) < p
        m[flip] = ~m[flip]
        return m


# ============================
# Ejemplo de uso
# ============================
if __name__ == "__main__":
    # NOTA: Usa tu CSV. Aquí dejo 'diabetes.csv' como en tu ejemplo.
    df = pd.read_csv('diabetes.csv')
    X = df.drop('target', axis=1).values
    y = df['target'].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    baseline = LinearRegression().fit(X_train_scaled, y_train)
    baseline_mae = mean_absolute_error(y_test, baseline.predict(X_test_scaled))
    baseline_mse = np.mean((y_test - baseline.predict(X_test_scaled)) ** 2)
    print(f"Baseline - MAE: {baseline_mae:.4f}")

    # GP + GA
    gp_optimizer = EvolutionaryOptimizer(
        maxtime=1200,                 # total
        # GP
        population_size=50,
        n_features_to_create=8,
        max_depth=5,
        complexity_penalty=1e-3,
        # GA (feature selection)
        do_feature_selection=True,
        fs_time_frac=0.3,             # 30% del tiempo para GA
        fs_population_size=60,
        fs_l1_penalty=1e-4,           # ajusta este lambda si quieres menos/más columnas
        fs_cv=3
    )

    gp_optimizer.fit(X_train, y_train)
    y_pred = gp_optimizer.predict(X_test)
    gp_mae = mean_absolute_error(y_test, y_pred)
    gp_mse = np.mean((y_test - y_pred) ** 2)
    print("\nResultados:")
    print(f"Baseline: MAE={baseline_mae:.4f}")
    print(f"Baseline: MSE={baseline_mse:.4f}")
    print(f"GP+GA:    MAE={gp_mae:.4f}")
    print(f"Mejora:   {((baseline_mae - gp_mae) / baseline_mae * 100):+.2f}%")
    print(f"GP+GA:    MSE={gp_mse:.4f}")
    print(f"Mejora:   {((baseline_mse - gp_mse) / baseline_mse * 100):+.2f}%")
    
    # Evaluación individual de árboles (opcional)
    #print(f"\n{'='*60}")
    #print(f"EVALUACIÓN INDIVIDUAL DE LOS {len(gp_optimizer.best_trees_)} ÁRBOLES")
    #print(f"{'='*60}")
    #for i, tree in enumerate(gp_optimizer.best_trees_, 1):
    #    try:
    #        # Construir datasets con SOLO este árbol añadido
    #        tree_train = np.clip(
    #            np.nan_to_num(tree.evaluate(scaler.transform(X_train)), nan=0.0, posinf=100.0, neginf=-100.0),
    #            -1000, 1000
    #        )
    #        tree_test = np.clip(
    #            np.nan_to_num(tree.evaluate(scaler.transform(X_test)), nan=0.0, posinf=100.0, neginf=-100.0),
    #            -1000, 1000
    #        )
    #        Xtr = np.column_stack([X_train_scaled, tree_train])
    #        Xte = np.column_stack([X_test_scaled, tree_test])
#
    #        model = LinearRegression().fit(Xtr, y_train)
    #        pred = model.predict(Xte)
    #        mae = mean_absolute_error(y_test, pred)
    #        imp = ((baseline_mae - mae) / baseline_mae * 100)
    #        print(f"Árbol {i}: {tree.to_string()}")
    #        print(f"  MAE: {mae:.4f} | Mejora: {imp:+.2f}% | Tamaño: {tree.size()} | Prof: {tree.depth()}\n")
    #    except Exception as e:
    #        print(f"Árbol {i}: {tree.to_string()}\n  ERROR: {e}\n")
