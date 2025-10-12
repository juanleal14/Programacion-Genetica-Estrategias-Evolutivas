# CELDA ADICIONAL PARA EL NOTEBOOK - Copia y pega este código

# Importar las funciones y librerías adicionales
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.linear_model import Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import clone
import pandas as pd

# Modelos adicionales para probar
additional_models = {
    'Lasso': Lasso(alpha=1.0, random_state=42),
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
    'BayesianRidge': BayesianRidge(),
    'HuberRegressor': HuberRegressor(),
    'KNeighbors': KNeighborsRegressor(n_neighbors=5),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
    'Bagging': BaggingRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'MLP': MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    'KernelRidge': KernelRidge(alpha=1.0)
}

print(f"\n{'='*80}")
print("PROBANDO MODELOS ADICIONALES")
print(f"{'='*80}")

results = []

for name, model in additional_models.items():
    try:
        print(f"\nProbando {name}...")
        
        # Baseline
        model_baseline = clone(model)
        model_baseline.fit(X_train, y_train)
        baseline_preds = model_baseline.predict(X_test)
        
        baseline_mae = mean_absolute_error(y_test, baseline_preds)
        baseline_mse = mean_squared_error(y_test, baseline_preds)
        
        # Con optimización
        model_optimized = clone(model)
        model_optimized.fit(X_train_optimized, y_train)
        optimized_preds = model_optimized.predict(X_test_optimized)
        
        optimized_mae = mean_absolute_error(y_test, optimized_preds)
        optimized_mse = mean_squared_error(y_test, optimized_preds)
        
        # Mejoras
        mae_improvement = ((baseline_mae - optimized_mae) / baseline_mae * 100)
        mse_improvement = ((baseline_mse - optimized_mse) / baseline_mse * 100)
        
        results.append({
            'Modelo': name,
            'MAE_Base': baseline_mae,
            'MAE_Opt': optimized_mae,
            'Mejora_MAE': mae_improvement,
            'MSE_Base': baseline_mse,
            'MSE_Opt': optimized_mse,
            'Mejora_MSE': mse_improvement
        })
        
        print(f"  MAE: {baseline_mae:.4f} → {optimized_mae:.4f} ({mae_improvement:+.2f}%)")
        print(f"  MSE: {baseline_mse:.4f} → {optimized_mse:.4f} ({mse_improvement:+.2f}%)")
        
    except Exception as e:
        print(f"  Error: {e}")
        continue

# Mostrar resumen
print(f"\n{'='*100}")
print("RESUMEN COMPLETO - TODOS LOS MODELOS")
print(f"{'='*100}")

df_results = pd.DataFrame(results)
df_sorted = df_results.sort_values('Mejora_MSE', ascending=False)

print(f"{'Modelo':<15} {'MAE Base':<10} {'MAE Opt':<10} {'Mejora MAE':<12} {'MSE Base':<10} {'MSE Opt':<10} {'Mejora MSE':<12}")
print("-" * 100)

for _, row in df_sorted.iterrows():
    print(f"{row['Modelo']:<15} {row['MAE_Base']:<10.4f} {row['MAE_Opt']:<10.4f} "
          f"{row['Mejora_MAE']:+<12.2f}% {row['MSE_Base']:<10.4f} {row['MSE_Opt']:<10.4f} "
          f"{row['Mejora_MSE']:+<12.2f}%")

# Estadísticas finales
print(f"\n{'='*60}")
print("ESTADÍSTICAS GENERALES")
print(f"{'='*60}")
print(f"Modelos que mejoraron MAE: {len(df_sorted[df_sorted['Mejora_MAE'] > 0])}/{len(df_sorted)}")
print(f"Modelos que mejoraron MSE: {len(df_sorted[df_sorted['Mejora_MSE'] > 0])}/{len(df_sorted)}")
print(f"Mejor mejora MAE: {df_sorted['Mejora_MAE'].max():.2f}% ({df_sorted.loc[df_sorted['Mejora_MAE'].idxmax(), 'Modelo']})")
print(f"Mejor mejora MSE: {df_sorted['Mejora_MSE'].max():.2f}% ({df_sorted.loc[df_sorted['Mejora_MSE'].idxmax(), 'Modelo']})")
print(f"Mejora promedio MAE: {df_sorted['Mejora_MAE'].mean():.2f}%")
print(f"Mejora promedio MSE: {df_sorted['Mejora_MSE'].mean():.2f}%")