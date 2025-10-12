# MODELOS AVANZADOS (requieren instalación adicional)
# pip install lightgbm catboost

# Descomenta y usa si tienes estas librerías instaladas:

"""
# LightGBM y CatBoost
try:
    import lightgbm as lgb
    import catboost as cb
    
    advanced_models = {
        'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1, n_estimators=100),
        'CatBoost': cb.CatBoostRegressor(random_state=42, verbose=False, iterations=100)
    }
    
    print(f"\n{'='*80}")
    print("PROBANDO MODELOS AVANZADOS")
    print(f"{'='*80}")
    
    for name, model in advanced_models.items():
        try:
            print(f"\nProbando {name}...")
            
            # Baseline
            model_baseline = model
            model_baseline.fit(X_train, y_train)
            baseline_preds = model_baseline.predict(X_test)
            
            baseline_mae = mean_absolute_error(y_test, baseline_preds)
            baseline_mse = mean_squared_error(y_test, baseline_preds)
            
            # Con optimización
            from sklearn.base import clone
            model_optimized = clone(model)
            model_optimized.fit(X_train_optimized, y_train)
            optimized_preds = model_optimized.predict(X_test_optimized)
            
            optimized_mae = mean_absolute_error(y_test, optimized_preds)
            optimized_mse = mean_squared_error(y_test, optimized_preds)
            
            # Mejoras
            mae_improvement = ((baseline_mae - optimized_mae) / baseline_mae * 100)
            mse_improvement = ((baseline_mse - optimized_mse) / baseline_mse * 100)
            
            print(f"  MAE: {baseline_mae:.4f} → {optimized_mae:.4f} ({mae_improvement:+.2f}%)")
            print(f"  MSE: {baseline_mse:.4f} → {optimized_mse:.4f} ({mse_improvement:+.2f}%)")
            
        except Exception as e:
            print(f"  Error con {name}: {e}")
            
except ImportError:
    print("LightGBM o CatBoost no están instalados")
    print("Instala con: pip install lightgbm catboost")
"""

# MODELOS CON DIFERENTES HIPERPARÁMETROS
print(f"\n{'='*80}")
print("PROBANDO VARIACIONES DE HIPERPARÁMETROS")
print(f"{'='*80}")

# Ridge con diferentes alphas
ridge_alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
print("\nRidge con diferentes alphas:")
for alpha in ridge_alphas:
    model = Ridge(alpha=alpha, random_state=42)
    
    # Baseline
    model.fit(X_train, y_train)
    baseline_preds = model.predict(X_test)
    baseline_mse = mean_squared_error(y_test, baseline_preds)
    
    # Optimizado
    model_opt = Ridge(alpha=alpha, random_state=42)
    model_opt.fit(X_train_optimized, y_train)
    opt_preds = model_opt.predict(X_test_optimized)
    opt_mse = mean_squared_error(y_test, opt_preds)
    
    improvement = ((baseline_mse - opt_mse) / baseline_mse * 100)
    print(f"  Alpha {alpha}: {baseline_mse:.4f} → {opt_mse:.4f} ({improvement:+.2f}%)")

# Random Forest con diferentes n_estimators
rf_estimators = [50, 100, 200, 300]
print("\nRandom Forest con diferentes n_estimators:")
for n_est in rf_estimators:
    model = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1)
    
    # Baseline
    model.fit(X_train, y_train)
    baseline_preds = model.predict(X_test)
    baseline_mse = mean_squared_error(y_test, baseline_preds)
    
    # Optimizado
    model_opt = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1)
    model_opt.fit(X_train_optimized, y_train)
    opt_preds = model_opt.predict(X_test_optimized)
    opt_mse = mean_squared_error(y_test, opt_preds)
    
    improvement = ((baseline_mse - opt_mse) / baseline_mse * 100)
    print(f"  n_estimators {n_est}: {baseline_mse:.4f} → {opt_mse:.4f} ({improvement:+.2f}%)")

# KNN con diferentes k
knn_neighbors = [3, 5, 7, 10, 15]
print("\nKNN con diferentes k:")
for k in knn_neighbors:
    model = KNeighborsRegressor(n_neighbors=k)
    
    # Baseline
    model.fit(X_train, y_train)
    baseline_preds = model.predict(X_test)
    baseline_mse = mean_squared_error(y_test, baseline_preds)
    
    # Optimizado
    model_opt = KNeighborsRegressor(n_neighbors=k)
    model_opt.fit(X_train_optimized, y_train)
    opt_preds = model_opt.predict(X_test_optimized)
    opt_mse = mean_squared_error(y_test, opt_preds)
    
    improvement = ((baseline_mse - opt_mse) / baseline_mse * 100)
    print(f"  k={k}: {baseline_mse:.4f} → {opt_mse:.4f} ({improvement:+.2f}%)")

# MLP con diferentes arquitecturas
mlp_configs = [
    (50,), (100,), (200,),
    (50, 50), (100, 50), (100, 100)
]
print("\nMLP con diferentes arquitecturas:")
for config in mlp_configs:
    model = MLPRegressor(hidden_layer_sizes=config, max_iter=500, random_state=42)
    
    try:
        # Baseline
        model.fit(X_train, y_train)
        baseline_preds = model.predict(X_test)
        baseline_mse = mean_squared_error(y_test, baseline_preds)
        
        # Optimizado
        model_opt = MLPRegressor(hidden_layer_sizes=config, max_iter=500, random_state=42)
        model_opt.fit(X_train_optimized, y_train)
        opt_preds = model_opt.predict(X_test_optimized)
        opt_mse = mean_squared_error(y_test, opt_preds)
        
        improvement = ((baseline_mse - opt_mse) / baseline_mse * 100)
        print(f"  {config}: {baseline_mse:.4f} → {opt_mse:.4f} ({improvement:+.2f}%)")
    except:
        print(f"  {config}: Error en convergencia")

print(f"\n{'='*80}")
print("ANÁLISIS COMPLETADO")
print(f"{'='*80}")