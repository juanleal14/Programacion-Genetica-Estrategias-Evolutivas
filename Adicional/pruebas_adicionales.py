# Modelos adicionales para probar el rendimiento
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.linear_model import Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb
import catboost as cb

def test_additional_models(X_train, X_test, y_train, y_test, X_train_optimized, X_test_optimized):
    """Prueba modelos adicionales para comparar rendimiento"""
    
    models = {
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
        'KernelRidge': KernelRidge(alpha=1.0),
        'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
        'CatBoost': cb.CatBoostRegressor(random_state=42, verbose=False)
    }
    
    results = []
    
    for name, model in models.items():
        try:
            print(f"\n{'='*70}")
            print(f"PROBANDO {name.upper()}")
            print(f"{'='*70}")
            
            # Baseline
            model_baseline = model
            model_baseline.fit(X_train, y_train)
            baseline_preds = model_baseline.predict(X_test)
            
            baseline_mae = mean_absolute_error(y_test, baseline_preds)
            baseline_mse = mean_squared_error(y_test, baseline_preds)
            
            print(f"Baseline MAE: {baseline_mae:.4f}")
            print(f"Baseline MSE: {baseline_mse:.4f}")
            
            # Con optimización
            from sklearn.base import clone
            model_optimized = clone(model)
            model_optimized.fit(X_train_optimized, y_train)
            optimized_preds = model_optimized.predict(X_test_optimized)
            
            optimized_mae = mean_absolute_error(y_test, optimized_preds)
            optimized_mse = mean_squared_error(y_test, optimized_preds)
            
            print(f"Optimizado MAE: {optimized_mae:.4f}")
            print(f"Optimizado MSE: {optimized_mse:.4f}")
            
            # Mejoras
            mae_improvement = ((baseline_mae - optimized_mae) / baseline_mae * 100)
            mse_improvement = ((baseline_mse - optimized_mse) / baseline_mse * 100)
            
            print(f"Mejora MAE: {mae_improvement:+.2f}%")
            print(f"Mejora MSE: {mse_improvement:+.2f}%")
            
            results.append({
                'Model': name,
                'Baseline_MAE': baseline_mae,
                'Optimized_MAE': optimized_mae,
                'MAE_Improvement': mae_improvement,
                'Baseline_MSE': baseline_mse,
                'Optimized_MSE': optimized_mse,
                'MSE_Improvement': mse_improvement
            })
            
        except Exception as e:
            print(f"Error con {name}: {e}")
            continue
    
    return results

# Función para mostrar resumen de resultados
def show_results_summary(results):
    """Muestra resumen ordenado de resultados"""
    import pandas as pd
    
    df_results = pd.DataFrame(results)
    
    print(f"\n{'='*100}")
    print("RESUMEN DE RESULTADOS - ORDENADO POR MEJORA EN MSE")
    print(f"{'='*100}")
    
    # Ordenar por mejora en MSE (descendente)
    df_sorted = df_results.sort_values('MSE_Improvement', ascending=False)
    
    print(f"{'Modelo':<15} {'MAE Base':<10} {'MAE Opt':<10} {'Mejora MAE':<12} {'MSE Base':<10} {'MSE Opt':<10} {'Mejora MSE':<12}")
    print("-" * 100)
    
    for _, row in df_sorted.iterrows():
        print(f"{row['Model']:<15} {row['Baseline_MAE']:<10.4f} {row['Optimized_MAE']:<10.4f} "
              f"{row['MAE_Improvement']:+<12.2f}% {row['Baseline_MSE']:<10.4f} {row['Optimized_MSE']:<10.4f} "
              f"{row['MSE_Improvement']:+<12.2f}%")
    
    # Estadísticas
    print(f"\n{'='*50}")
    print("ESTADÍSTICAS")
    print(f"{'='*50}")
    print(f"Modelos que mejoraron MAE: {len(df_sorted[df_sorted['MAE_Improvement'] > 0])}/{len(df_sorted)}")
    print(f"Modelos que mejoraron MSE: {len(df_sorted[df_sorted['MSE_Improvement'] > 0])}/{len(df_sorted)}")
    print(f"Mejor mejora MAE: {df_sorted['MAE_Improvement'].max():.2f}% ({df_sorted.loc[df_sorted['MAE_Improvement'].idxmax(), 'Model']})")
    print(f"Mejor mejora MSE: {df_sorted['MSE_Improvement'].max():.2f}% ({df_sorted.loc[df_sorted['MSE_Improvement'].idxmax(), 'Model']})")
    print(f"Mejora promedio MAE: {df_sorted['MAE_Improvement'].mean():.2f}%")
    print(f"Mejora promedio MSE: {df_sorted['MSE_Improvement'].mean():.2f}%")