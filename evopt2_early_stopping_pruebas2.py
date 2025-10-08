import numpy as np
import pandas as pd
import time
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')


# ============================
# Utilidades de métricas
# ============================
def cv_mae_mse(model, X, y, n_splits=3, random_state=42):
    """Devuelve (MAE_media, MSE_media) con KFold shuffle=True."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    maes, mses = [], []
    for tr, va in kf.split(X):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        model.fit(Xtr, ytr)
        pred = model.predict(Xva)
        maes.append(mean_absolute_error(yva, pred))
        mses.append(np.mean((yva - pred) ** 2))
    return float(np.mean(maes)), float(np.mean(mses))


def combine_fitness(mae, mse, mode="combo", w_mae=1.0, w_mse=0.001):
    """Convierte (MAE, MSE) en un escalar de fitness."""
    if mode == "mae":
        return mae
    if mode == "mse":
        return mse
    # combo
    return w_mae * mae + w_mse * mse


class GPNode:
    """Nodo de árbol de programación genética."""
    def __init__(self, value, children=None, node_type='terminal'):
        self.value = value
        self.children = children or []
        self.node_type = node_type  # 'terminal' o 'function'

    def evaluate(self, X):
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
        return GPNode(self.value, [child.copy() for child in self.children], self.node_type)

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
    GP (síntesis) + GA (selección). Incluye early stopping, restarts, parsimonia adaptativa, fast gate.
    Fitness configurable: 'mae', 'mse' o 'combo' (MAE + λ*MSE).
    """
    def __init__(self,
                 # — tiempo —
                 maxtime=1200,  # 20 minutos
                 # — GP —
                 population_size=50,
                 n_features_to_create=8,
                 mutation_prob=0.25,
                 crossover_prob=0.8,
                 tournament_size=5,
                 max_depth=5,
                 elite_frac=0.1,
                 complexity_penalty=5e-4,
                 # — early / restarts —
                 early_stop_patience=120,
                 early_stop_min_improv=1e-5,
                 restart_on_stagnation=True,
                 restart_frac=0.20,
                 adaptive_parsimony=True,
                 parsimony_boost=1.5,
                 use_fast_gate=True,
                 gate_keep_top=0.6,
                 # — fitness (GP y GA) —
                 fitness_mode="combo",  # 'mae' | 'mse' | 'combo'
                 fitness_w_mae=1.0,
                 fitness_w_mse=0.001,
                 # — GA (selección) —
                 do_feature_selection=True,
                 fs_time_frac=0.25,
                 fs_population_size=60,
                 fs_tournament_size=4,
                 fs_crossover_prob=0.9,
                 fs_mutation_prob=0.15,
                 fs_elite_frac=0.1,
                 fs_l1_penalty=1e-3,
                 fs_cv=3,
                 keep_originals=None  # True = originales obligatorias; False = originales pueden caer; None = libre
                 ):
        # tiempo
        self.maxtime = maxtime

        # GP
        self.population_size = population_size
        self.n_features_to_create = n_features_to_create
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        self.elite_size = max(1, int(population_size * elite_frac))
        self.complexity_penalty = complexity_penalty

        self.functions = {
            'add': 2, 'sub': 2, 'mul': 2, 'div': 2,
            'sqrt': 1, 'square': 1, 'log': 1, 'abs': 1, 'sin': 1, 'cos': 1, 'tanh': 1
        }

        self.best_trees_ = []
        self.best_fitness_ = float('inf')
        self.fitness_history_ = []

        # early / restart
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_improv = early_stop_min_improv
        self.restart_on_stagnation = restart_on_stagnation
        self.restart_frac = restart_frac
        self.adaptive_parsimony = adaptive_parsimony
        self.parsimony_boost = parsimony_boost
        self.use_fast_gate = use_fast_gate
        self.gate_keep_top = gate_keep_top

        # fitness
        self.fitness_mode = fitness_mode
        self.fitness_w_mae = fitness_w_mae
        self.fitness_w_mse = fitness_w_mse

        # GA
        self.do_feature_selection = do_feature_selection
        self.fs_time_frac = fs_time_frac
        self.fs_population_size = fs_population_size
        self.fs_tournament_size = fs_tournament_size
        self.fs_crossover_prob = fs_crossover_prob
        self.fs_mutation_prob = fs_mutation_prob
        self.fs_elite_size = max(1, int(fs_population_size * fs_elite_frac))
        self.fs_l1_penalty = fs_l1_penalty
        self.fs_cv = fs_cv
        self.keep_originals = keep_originals

        # salidas
        self.selected_mask_ = None
        self.model_ = None
        self.scaler_ = None
        self.n_features_in_ = None

    # ============================
    # API
    # ============================
    def fit(self, X, y):
        start_time = time.time()

        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.n_features_in_ = X.shape[1]

        gp_budget = self.maxtime * (1.0 - (self.fs_time_frac if self.do_feature_selection else 0.0))
        ga_budget = self.maxtime - gp_budget

        print("\n" + "="*70)
        print("FASE 1 — PROGRAMACIÓN GENÉTICA (síntesis de atributos)")
        print("="*70)
        print(f"Población: {self.population_size} | Nuevas features por individuo: {self.n_features_to_create}")
        print(f"Profundidad máxima: {self.max_depth} | Tiempo GP: {gp_budget/60:.1f} min")
        print("="*70 + "\n")

        population = [self._create_random_individual() for _ in range(self.population_size)]

        # Semilla inicial (mejor de la población inicial por fast fit)
        init_fits_fast = [self._evaluate_individual_gp(ind, X_scaled, y, fast=True) for ind in population]
        init_idx = int(np.argmin(init_fits_fast))
        self.best_trees_ = [t.copy() for t in population[init_idx]]
        self.best_fitness_ = float(self._evaluate_individual_gp(self.best_trees_, X_scaled, y, fast=False))
        print(f"Seed inicial - Fitness (CV): {self.best_fitness_:.6f}")

        generation = 0
        no_improve = 0
        base_complexity_penalty = self.complexity_penalty

        while (time.time() - start_time) < gp_budget:
            generation += 1

            # Fast gate
            if self.use_fast_gate:
                fast_fits = [self._evaluate_individual_gp(ind, X_scaled, y, fast=True) for ind in population]
                k = max(1, int(self.gate_keep_top * len(population)))
                top_idx = np.argsort(fast_fits)[:k]
                fitness_scores = [None] * len(population)
                for i in top_idx:
                    fitness_scores[i] = self._evaluate_individual_gp(population[i], X_scaled, y, fast=False)
                for i in range(len(population)):
                    if fitness_scores[i] is None:
                        fitness_scores[i] = fast_fits[i]
            else:
                fitness_scores = [self._evaluate_individual_gp(ind, X_scaled, y, fast=False) for ind in population]

            best_idx = int(np.argmin(fitness_scores))
            best_fit_gen = float(fitness_scores[best_idx])

            is_first = np.isinf(self.best_fitness_) or np.isnan(self.best_fitness_)
            improv_threshold = self.early_stop_min_improv * (0.0 if is_first else max(1.0, abs(self.best_fitness_)))

            if is_first or (best_fit_gen < self.best_fitness_ - improv_threshold):
                self.best_fitness_ = best_fit_gen
                self.best_trees_ = [t.copy() for t in population[best_idx]]
                print(f"Gen {generation} - MEJORA! Fitness: {self.best_fitness_:.6f}")
                if self.best_trees_:
                    print(f"  Árbol ejemplo: {self.best_trees_[0].to_string()}")
                no_improve = 0
                self.complexity_penalty = base_complexity_penalty
            else:
                no_improve += 1
                if self.adaptive_parsimony and (no_improve % max(1, self.early_stop_patience // 2) == 0):
                    self.complexity_penalty *= self.parsimony_boost
                    print(f"  [Parsimonia↑] complexity_penalty → {self.complexity_penalty:.2e}")

            self.fitness_history_.append(self.best_fitness_)

            # early stop / restart
            if no_improve >= self.early_stop_patience:
                if self.restart_on_stagnation:
                    n_restart = max(1, int(self.restart_frac * len(population)))
                    print(f"  [Restart] Estancamiento {no_improve} gens → resembrando {n_restart} individuos")
                    for _ in range(n_restart):
                        j = random.randrange(len(population))
                        population[j] = self._create_random_individual()
                    no_improve = 0
                    self.complexity_penalty = base_complexity_penalty
                else:
                    print("  [EarlyStop] Paro por estancamiento en GP.")
                    break

            # reproducción
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

        # Backup por si algo raro dejó vacío
        if not self.best_trees_:
            try:
                quick_fits = [self._evaluate_individual_gp(ind, X_scaled, y, fast=True) for ind in population]
                backup_idx = int(np.argmin(quick_fits))
            except Exception:
                backup_idx = 0
            self.best_trees_ = [t.copy() for t in population[backup_idx]]
            self.best_fitness_ = float(self._evaluate_individual_gp(self.best_trees_, X_scaled, y, fast=False))
            print("  [Backup] best_trees_ se inicializó con el mejor individuo disponible.")

        # Dataset aumentado (train) con mejores árboles
        X_train_aug = self._augment_with_trees(X_scaled, self.best_trees_)
        # Modelo provisional (todas columnas)
        self.model_ = LinearRegression().fit(X_train_aug, y)
        self.selected_mask_ = np.ones(X_train_aug.shape[1], dtype=bool)

        # ---------- FASE 2: GA ----------
        elapsed = time.time() - start_time
        if self.do_feature_selection and elapsed < self.maxtime:
            remaining = self.maxtime - elapsed
            ga_budget = min(ga_budget, remaining)
            print("\n" + "="*70)
            print("FASE 2 — GA de Selección de Features")
            print("="*70)
            print(f"Población GA: {self.fs_population_size} | Tiempo GA: {ga_budget/60:.1f} min")
            print(f"Penalización L1 (nº columnas): {self.fs_l1_penalty}")
            print("="*70 + "\n")

            # referencia (CV) con todas las columnas usando Ridge
            ref_mae, ref_mse = cv_mae_mse(Ridge(alpha=1.0, random_state=42),
                                          X_train_aug, y, n_splits=self.fs_cv, random_state=42)
            ref_fit = combine_fitness(ref_mae, ref_mse,
                                      mode=self.fitness_mode,
                                      w_mae=self.fitness_w_mae,
                                      w_mse=self.fitness_w_mse)

            mask, model = self._run_feature_selection_ga(X_train_aug, y, time_budget=ga_budget)
            if mask is not None and model is not None:
                # comprobar si GA mejora la referencia
                X_sub = X_train_aug[:, mask]
                fs_mae, fs_mse = cv_mae_mse(Ridge(alpha=1.0, random_state=42),
                                            X_sub, y, n_splits=self.fs_cv, random_state=42)
                fs_fit = combine_fitness(fs_mae, fs_mse,
                                         mode=self.fitness_mode,
                                         w_mae=self.fitness_w_mae,
                                         w_mse=self.fitness_w_mse)

                if fs_fit <= ref_fit + 1e-9:
                    self.selected_mask_ = mask
                    self.model_ = model  # ya entrenado en _fs_fitness
                    print(f"Selección GA: {self.selected_mask_.sum()}/{len(self.selected_mask_)} columnas. "
                          f"(CV fit {fs_fit:.4f} <= ref {ref_fit:.4f})")
                else:
                    print(f"Selección GA RECHAZADA: (CV fit {fs_fit:.4f} > ref {ref_fit:.4f}). "
                          "Se mantiene el conjunto completo de GP.")
                    self.selected_mask_ = np.ones(X_train_aug.shape[1], dtype=bool)
                    self.model_ = Ridge(alpha=1.0, random_state=42).fit(X_train_aug, y)

        print("\nEntrenamiento completado.")
        return self

    def transform(self, X):
        if hasattr(X, 'values'):
            X = X.values
        X_scaled = self.scaler_.transform(X)
        trees = self.best_trees_ if isinstance(self.best_trees_, list) else []
        X_aug = self._augment_with_trees(X_scaled, trees) if trees else X_scaled.copy()

        if self.selected_mask_ is None:
            return X_aug
        if self.selected_mask_.shape[0] != X_aug.shape[1]:
            m = min(self.selected_mask_.shape[0], X_aug.shape[1])
            return X_aug[:, self.selected_mask_[:m]]
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

    def _evaluate_individual_gp(self, trees, X, y, fast=False):
        try:
            X_new = self._augment_with_trees(X, trees)
            if np.any(np.abs(X_new) > 1e6) or np.any(np.std(X_new, axis=0) < 1e-10):
                return 1e6
            model = LinearRegression()
            if fast:
                X_tr, X_va, y_tr, y_va = train_test_split(X_new, y, test_size=0.25, random_state=42)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_va)
                mae = mean_absolute_error(y_va, pred)
                mse = np.mean((y_va - pred) ** 2)
            else:
                # CV con LR para GP (más rápido); fitness puede ser combo
                mae, mse = cv_mae_mse(model, X_new, y, n_splits=3, random_state=42)
            complexity = sum(t.size() for t in trees)
            penalty = self.complexity_penalty * complexity
            fit = combine_fitness(mae, mse, mode=self.fitness_mode,
                                  w_mae=self.fitness_w_mae, w_mse=self.fitness_w_mse) + penalty
            # guardar modelo si mejora (solo versión no fast)
            if (not fast) and (fit < self.best_fitness_):
                model.fit(X_new, y)
                self.model_ = model
            return fit
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
        n_swap = random.randint(1, max(1, min(len(c1), len(c2)) // 2))
        indices = random.sample(range(min(len(c1), len(c2))), n_swap)
        for idx in indices:
            c1[idx], c2[idx] = c2[idx].copy(), c1[idx].copy()
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
        for j, child in enumerate(m.children):
            if random.random() < 0.2:
                m.children[j] = self._mutate_single_tree(child)
        return m

    # ============================
    # GA de selección de features
    # ============================
    def _run_feature_selection_ga(self, X_aug, y, time_budget):
        start = time.time()
        d = X_aug.shape[1]
        if d == 0:
            return None, None

        n_orig = self.n_features_in_

        pop = []
        for _ in range(self.fs_population_size):
            mask = np.zeros(d, dtype=bool)
            base_orig = np.random.rand(n_orig) < (0.7 if self.keep_originals is not False else 0.5)
            base_synth = np.random.rand(d - n_orig) < 0.4
            mask[:n_orig] = base_orig
            mask[n_orig:] = base_synth
            if self.keep_originals is True:
                mask[:n_orig] = True
            if not mask.any():
                mask[np.random.randint(0, d)] = True
            pop.append(mask)

        best_mask = None
        best_fit = np.inf
        best_model = None
        gen = 0

        while (time.time() - start) < time_budget:
            gen += 1
            fits, models = [], []
            for mask in pop:
                fit, model = self._fs_fitness(X_aug, y, mask)
                fits.append(fit)
                models.append(model)
                if fit < best_fit:
                    best_fit = fit
                    best_mask = mask.copy()
                    best_model = model
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
                if self.keep_originals is True:
                    c1[:n_orig] = True
                    c2[:n_orig] = True
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
        if not mask.any():
            return 1e9, None
        X_sub = X_aug[:, mask]
        if np.any(np.std(X_sub, axis=0) < 1e-12):
            return 1e8, None

        # REG: Ridge para estabilidad
        model = Ridge(alpha=1.0, random_state=42)
        mae, mse = cv_mae_mse(model, X_sub, y, n_splits=self.fs_cv, random_state=42)
        l1 = self.fs_l1_penalty * mask.sum()
        fitness = combine_fitness(mae, mse, mode=self.fitness_mode,
                                  w_mae=self.fitness_w_mae, w_mse=self.fitness_w_mse) + l1
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
    # Utilidad
    # ============================
    def selected_feature_info(self):
        if self.selected_mask_ is None:
            raise ValueError("Aún no hay selección (selected_mask_).")
        info = []
        n_orig = self.n_features_in_
        idxs = np.where(self.selected_mask_)[0]
        for j in idxs:
            if j < n_orig:
                info.append((j, f"X{j} (original)"))
            else:
                k = j - n_orig
                if 0 <= k < len(self.best_trees_):
                    info.append((j, f"{self.best_trees_[k].to_string()} (creada)"))
                else:
                    info.append((j, f"Z{k} (creada)"))
        return info


# ============================
# Ejemplo de uso
# ============================
if __name__ == "__main__":
    df = pd.read_csv('diabetes.csv')
    X = df.drop('target', axis=1).values
    y = df['target'].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline
    base_scaler = RobustScaler()
    X_train_scaled = base_scaler.fit_transform(X_train)
    X_test_scaled = base_scaler.transform(X_test)

    lr_base = LinearRegression().fit(X_train_scaled, y_train)
    base_pred = lr_base.predict(X_test_scaled)
    base_mae = mean_absolute_error(y_test, base_pred)
    base_mse = np.mean((y_test - base_pred) ** 2)
    print(f"Baseline - MAE: {base_mae:.4f}")
    print(f"Baseline - MSE: {base_mse:.4f}")

    # GP + GA (20 min, combo MAE+MSE)
    gp_optimizer = EvolutionaryOptimizer(
        maxtime=1200,                 # 20 minutos
        # GP
        population_size=60,
        n_features_to_create=8,
        max_depth=5,
        complexity_penalty=5e-4,
        mutation_prob=0.25,
        crossover_prob=0.8,
        early_stop_patience=120,
        early_stop_min_improv=1e-5,
        restart_on_stagnation=True,
        restart_frac=0.20,
        adaptive_parsimony=True,
        parsimony_boost=1.5,
        use_fast_gate=True,
        gate_keep_top=0.6,
        # Fitness (usa ambos)
        fitness_mode="mae",  # 'mae' | 'mse' | 'combo'
        fitness_w_mae=1.0,
        fitness_w_mse=0.001,   # ajusta si el MSE pesa demasiado/poco
        # GA
        do_feature_selection=True,
        fs_time_frac=0.25,
        fs_population_size=60,
        fs_l1_penalty=1e-3,
        fs_mutation_prob=0.15,
        fs_crossover_prob=0.9,
        fs_cv=3,
        keep_originals=None
    )

    gp_optimizer.fit(X_train, y_train)
    y_pred = gp_optimizer.predict(X_test)

    gp_mae = mean_absolute_error(y_test, y_pred)
    gp_mse = np.mean((y_test - y_pred) ** 2)

    print("\nResultados:")
    print(f"Baseline: MAE={base_mae:.4f} | MSE={base_mse:.4f}")
    print(f"GP+GA:    MAE={gp_mae:.4f} | MSE={gp_mse:.4f}")
    print(f"Mejora MAE: {((base_mae - gp_mae) / base_mae * 100):+.2f}%")
    print(f"Mejora MSE: {((base_mse - gp_mse) / base_mse * 100):+.2f}%")

    print("\nFeatures seleccionadas:")
    for idx, desc in gp_optimizer.selected_feature_info():
        print(f"  [{idx}] {desc}")
