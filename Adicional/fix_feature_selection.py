# SOLUCIÓN PARA EL PROBLEMA DE FEATURE SELECTION

# Cambiar estos parámetros en el método _evolutionary_feature_selection_cv:

def _evolutionary_feature_selection_cv(self, X_full, y_full, 
                                       population_size=50, max_time=None):  # Aumentar población
    """Selección evolutiva de features usando Cross-Validation con límite de tiempo."""
    n_features = X_full.shape[1]
    
    # División train/validation para FS
    n_val = int(len(X_full) * 0.2)
    indices = np.random.permutation(len(X_full))
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    
    X_fs_train, X_fs_val = X_full[train_idx], X_full[val_idx]
    y_fs_train, y_fs_val = y_full[train_idx], y_full[val_idx]
    
    # Inicializar población con MÁS DIVERSIDAD
    population = []
    for _ in range(population_size):
        individual = np.zeros(n_features, dtype=bool)
        # Variar más el número de features seleccionadas
        n_selected = random.randint(2, min(n_features-1, max(8, n_features//2)))
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
        
        # CAMBIO PRINCIPAL: Early stopping más permisivo O usar tiempo completo
        max_early_stop = max(100, population_size * 2)  # Mínimo 100 generaciones
        if fs_early_stop >= max_early_stop:
            print(f"  FS Early stopping en generación {gen}")
            break
        
        # Nueva generación basada en validación
        new_population = []
        
        elite_size = max(2, population_size // 8)  # Más elite
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
            
            if random.random() < 0.4:  # Más mutación
                child1 = self._mutate_fs(child1)
            if random.random() < 0.4:
                child2 = self._mutate_fs(child2)
            
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]
    
    print(f"  Selección completada en {gen} generaciones")
    return best_individual, best_metrics

# ALTERNATIVA: Cambiar la llamada en fit() para usar más tiempo en FS
# En lugar de:
# fs_time = total_time * 0.3  # 30% para Feature Selection

# Usar:
# fs_time = total_time * 0.5  # 50% para Feature Selection
# gp_time = total_time * 0.5  # 50% para GP

# O incluso forzar un mínimo de tiempo:
# fs_time = max(total_time * 0.3, 300)  # Mínimo 5 minutos para FS