# ANÁLISIS COMPLETO - CELDA FINAL PARA EL NOTEBOOK

print(f"\n{'='*100}")
print("ANÁLISIS COMPLETO DE RENDIMIENTO")
print(f"{'='*100}")

# Recopilar todos los resultados anteriores
all_results = []

# Modelos ya probados (añadir manualmente los resultados que ya tienes)
existing_results = [
    {'Modelo': 'Ridge', 'Mejora_MAE': 11.09, 'Mejora_MSE': 17.61},
    {'Modelo': 'LinearRegression', 'Mejora_MAE': 9.74, 'Mejora_MSE': 15.66},
    {'Modelo': 'RandomForest', 'Mejora_MAE': -1.47, 'Mejora_MSE': -3.11},
    {'Modelo': 'SVR', 'Mejora_MAE': -1.83, 'Mejora_MSE': 0.60},
    {'Modelo': 'XGBoost', 'Mejora_MAE': 2.44, 'Mejora_MSE': -0.33},
    {'Modelo': 'GradientBoosting', 'Mejora_MAE': 1.07, 'Mejora_MSE': 1.52}
]

# Probar modelos adicionales rápidamente
additional_quick_models = {
    'Lasso': Lasso(alpha=1.0, random_state=42),
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
    'BayesianRidge': BayesianRidge(),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    'AdaBoost': AdaBoostRegressor(n_estimators=50, random_state=42),
    'KNeighbors': KNeighborsRegressor(n_neighbors=5)
}

print("Probando modelos adicionales...")
for name, model in additional_quick_models.items():
    try:
        # Baseline
        model.fit(X_train, y_train)
        baseline_preds = model.predict(X_test)
        baseline_mae = mean_absolute_error(y_test, baseline_preds)
        baseline_mse = mean_squared_error(y_test, baseline_preds)
        
        # Optimizado
        from sklearn.base import clone
        model_opt = clone(model)
        model_opt.fit(X_train_optimized, y_train)
        opt_preds = model_opt.predict(X_test_optimized)
        opt_mae = mean_absolute_error(y_test, opt_preds)
        opt_mse = mean_squared_error(y_test, opt_preds)
        
        # Mejoras
        mae_improvement = ((baseline_mae - opt_mae) / baseline_mae * 100)
        mse_improvement = ((baseline_mse - opt_mse) / baseline_mse * 100)
        
        existing_results.append({
            'Modelo': name,
            'Mejora_MAE': mae_improvement,
            'Mejora_MSE': mse_improvement
        })
        
    except Exception as e:
        print(f"Error con {name}: {e}")

# Crear DataFrame y ordenar
import pandas as pd
df_final = pd.DataFrame(existing_results)
df_final = df_final.sort_values('Mejora_MSE', ascending=False)

print(f"\n{'='*80}")
print("RANKING FINAL DE MODELOS")
print(f"{'='*80}")
print(f"{'Posición':<8} {'Modelo':<18} {'Mejora MAE':<12} {'Mejora MSE':<12} {'Categoría':<15}")
print("-" * 80)

for i, (_, row) in enumerate(df_final.iterrows(), 1):
    # Categorizar mejora
    if row['Mejora_MSE'] > 10:
        categoria = "Excelente"
    elif row['Mejora_MSE'] > 5:
        categoria = "Muy Buena"
    elif row['Mejora_MSE'] > 0:
        categoria = "Buena"
    elif row['Mejora_MSE'] > -2:
        categoria = "Neutral"
    else:
        categoria = "Negativa"
    
    print(f"{i:<8} {row['Modelo']:<18} {row['Mejora_MAE']:+<12.2f}% {row['Mejora_MSE']:+<12.2f}% {categoria:<15}")

# Análisis estadístico
print(f"\n{'='*80}")
print("ANÁLISIS ESTADÍSTICO")
print(f"{'='*80}")

mejoras_positivas_mae = len(df_final[df_final['Mejora_MAE'] > 0])
mejoras_positivas_mse = len(df_final[df_final['Mejora_MSE'] > 0])
total_modelos = len(df_final)

print(f"Total de modelos probados: {total_modelos}")
print(f"Modelos con mejora en MAE: {mejoras_positivas_mae}/{total_modelos} ({mejoras_positivas_mae/total_modelos*100:.1f}%)")
print(f"Modelos con mejora en MSE: {mejoras_positivas_mse}/{total_modelos} ({mejoras_positivas_mse/total_modelos*100:.1f}%)")

print(f"\nMejor modelo (MSE): {df_final.iloc[0]['Modelo']} ({df_final.iloc[0]['Mejora_MSE']:+.2f}%)")
print(f"Peor modelo (MSE): {df_final.iloc[-1]['Modelo']} ({df_final.iloc[-1]['Mejora_MSE']:+.2f}%)")

print(f"\nMejora promedio MAE: {df_final['Mejora_MAE'].mean():+.2f}%")
print(f"Mejora promedio MSE: {df_final['Mejora_MSE'].mean():+.2f}%")

print(f"Desviación estándar MAE: {df_final['Mejora_MAE'].std():.2f}%")
print(f"Desviación estándar MSE: {df_final['Mejora_MSE'].std():.2f}%")

# Recomendaciones
print(f"\n{'='*80}")
print("RECOMENDACIONES")
print(f"{'='*80}")

top_3 = df_final.head(3)
print("🏆 TOP 3 MODELOS RECOMENDADOS:")
for i, (_, row) in enumerate(top_3.iterrows(), 1):
    print(f"{i}. {row['Modelo']}: {row['Mejora_MSE']:+.2f}% mejora en MSE")

# Análisis por tipo de modelo
linear_models = ['Ridge', 'LinearRegression', 'Lasso', 'ElasticNet', 'BayesianRidge']
tree_models = ['RandomForest', 'DecisionTree', 'ExtraTrees', 'XGBoost', 'GradientBoosting', 'AdaBoost']
other_models = ['SVR', 'KNeighbors']

print(f"\n📊 ANÁLISIS POR TIPO:")
for category, models in [("Modelos Lineales", linear_models), 
                        ("Modelos de Árbol", tree_models), 
                        ("Otros Modelos", other_models)]:
    category_data = df_final[df_final['Modelo'].isin(models)]
    if not category_data.empty:
        avg_improvement = category_data['Mejora_MSE'].mean()
        print(f"{category}: {avg_improvement:+.2f}% mejora promedio")

print(f"\n💡 CONCLUSIÓN:")
if df_final['Mejora_MSE'].mean() > 0:
    print("✅ La optimización evolutiva es EFECTIVA para este dataset")
    print(f"   Mejora promedio: {df_final['Mejora_MSE'].mean():+.2f}% en MSE")
else:
    print("⚠️  La optimización evolutiva tiene resultados mixtos")
    print("   Considera ajustar hiperparámetros o probar otros datasets")

print(f"\n🎯 MEJOR ESTRATEGIA:")
best_model = df_final.iloc[0]['Modelo']
best_improvement = df_final.iloc[0]['Mejora_MSE']
print(f"Usar {best_model} con las features optimizadas")
print(f"Esperada mejora: {best_improvement:+.2f}% en MSE")

print(f"\n{'='*100}")
print("ANÁLISIS COMPLETADO")
print(f"{'='*100}")