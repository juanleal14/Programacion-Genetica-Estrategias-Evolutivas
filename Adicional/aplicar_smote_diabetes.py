#!/usr/bin/env python3
"""
Script para aplicar SMOTE al dataset de diabetes y aumentar su tamaño.
SMOTE (Synthetic Minority Oversampling Technique) genera datos sintéticos
realistas basados en los datos existentes.
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def aplicar_smote_diabetes(archivo_original='diabetes.csv', 
                          archivo_ampliado='diabetes_smote.csv',
                          factor_aumento=50):
    """
    Aplica SMOTE al dataset de diabetes para aumentar su tamaño.
    
    Args:
        archivo_original: Nombre del archivo CSV original
        archivo_ampliado: Nombre del archivo CSV ampliado
        factor_aumento: Factor de aumento del dataset (50 = 50x más datos)
    """
    
    print("=" * 70)
    print("APLICANDO SMOTE AL DATASET DE DIABETES")
    print("=" * 70)
    
    # Cargar datos originales
    df_original = pd.read_csv(archivo_original)
    print(f"Dataset original: {df_original.shape[0]} filas, {df_original.shape[1]} columnas")
    
    # Separar features y target
    X = df_original.drop('target', axis=1)
    y = df_original['target']
    
    print(f"Features: {X.shape[1]}")
    print(f"Rango del target: {y.min():.1f} - {y.max():.1f}")
    
    # Para regresión, necesitamos convertir el target continuo en clases discretas
    # temporalmente para aplicar SMOTE
    print("\nDiscretizando target para SMOTE...")
    
    # Crear bins para el target (por ejemplo, 10 bins)
    n_bins = 10
    y_binned = pd.cut(y, bins=n_bins, labels=False)
    
    print(f"Target discretizado en {n_bins} clases")
    print(f"Distribución de clases:")
    for i in range(n_bins):
        count = (y_binned == i).sum()
        print(f"  Clase {i}: {count} muestras")
    
    # Calcular cuántas muestras necesitamos por clase
    tamaño_objetivo = df_original.shape[0] * factor_aumento
    muestras_por_clase = tamaño_objetivo // n_bins
    
    print(f"\nObjetivo: {tamaño_objetivo} muestras totales")
    print(f"Muestras por clase objetivo: {muestras_por_clase}")
    
    # Crear diccionario de sampling strategy
    sampling_strategy = {}
    for i in range(n_bins):
        current_count = (y_binned == i).sum()
        if current_count > 0:  # Solo si hay muestras en esta clase
            sampling_strategy[i] = max(muestras_por_clase, current_count)
    
    # Aplicar SMOTE
    print("\nAplicando SMOTE...")
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=42,
        k_neighbors=min(5, len(X) - 1)  # Ajustar k_neighbors si hay pocas muestras
    )
    
    try:
        X_resampled, y_binned_resampled = smote.fit_resample(X, y_binned)
        print(f"SMOTE aplicado exitosamente!")
        print(f"Nuevas dimensiones: {X_resampled.shape}")
        
        # Reconstruir el target continuo
        # Para cada muestra sintética, interpolar el valor del target
        # basado en la clase y añadir algo de ruido realista
        print("\nReconstruyendo target continuo...")
        
        # Calcular estadísticas por bin del target original
        bin_stats = []
        for i in range(n_bins):
            mask = y_binned == i
            if mask.sum() > 0:
                bin_values = y[mask]
                bin_stats.append({
                    'mean': bin_values.mean(),
                    'std': bin_values.std(),
                    'min': bin_values.min(),
                    'max': bin_values.max()
                })
            else:
                # Si no hay muestras en este bin, usar valores interpolados
                bin_stats.append({
                    'mean': y.mean(),
                    'std': y.std(),
                    'min': y.min(),
                    'max': y.max()
                })
        
        # Generar targets continuos para las muestras sintéticas
        y_resampled = []
        np.random.seed(42)
        
        for bin_class in y_binned_resampled:
            stats = bin_stats[bin_class]
            # Generar valor con distribución normal centrada en la media del bin
            synthetic_value = np.random.normal(stats['mean'], stats['std'] * 0.5)
            # Asegurar que esté dentro del rango del bin
            synthetic_value = np.clip(synthetic_value, stats['min'], stats['max'])
            y_resampled.append(synthetic_value)
        
        y_resampled = np.array(y_resampled)
        
        # Crear DataFrame ampliado
        df_ampliado = pd.DataFrame(X_resampled, columns=X.columns)
        df_ampliado['target'] = y_resampled
        
        # Guardar dataset ampliado
        df_ampliado.to_csv(archivo_ampliado, index=False)
        
        print("=" * 70)
        print("RESULTADOS")
        print("=" * 70)
        print(f"Dataset original: {df_original.shape[0]} filas")
        print(f"Dataset ampliado: {df_ampliado.shape[0]} filas")
        print(f"Factor de aumento: {df_ampliado.shape[0] / df_original.shape[0]:.1f}x")
        print(f"Archivo guardado: {archivo_ampliado}")
        
        # Estadísticas comparativas
        print("\nEstadísticas del target:")
        print("Original:")
        print(f"  Media: {y.mean():.2f}")
        print(f"  Std: {y.std():.2f}")
        print(f"  Min: {y.min():.2f}")
        print(f"  Max: {y.max():.2f}")
        
        print("Ampliado:")
        print(f"  Media: {y_resampled.mean():.2f}")
        print(f"  Std: {y_resampled.std():.2f}")
        print(f"  Min: {y_resampled.min():.2f}")
        print(f"  Max: {y_resampled.max():.2f}")
        
        # Crear visualización comparativa
        crear_visualizacion_comparativa(df_original, df_ampliado)
        
        return df_ampliado
        
    except Exception as e:
        print(f"Error al aplicar SMOTE: {e}")
        print("Intentando con parámetros más conservadores...")
        
        # Intentar con menos aumento
        factor_reducido = min(10, factor_aumento)
        tamaño_objetivo_reducido = df_original.shape[0] * factor_reducido
        muestras_por_clase_reducido = tamaño_objetivo_reducido // n_bins
        
        sampling_strategy_reducido = {}
        for i in range(n_bins):
            current_count = (y_binned == i).sum()
            if current_count > 0:
                sampling_strategy_reducido[i] = max(muestras_por_clase_reducido, current_count)
        
        smote_reducido = SMOTE(
            sampling_strategy=sampling_strategy_reducido,
            random_state=42,
            k_neighbors=min(3, len(X) - 1)
        )
        
        X_resampled, y_binned_resampled = smote_reducido.fit_resample(X, y_binned)
        
        # Reconstruir target (código similar al anterior)
        y_resampled = []
        np.random.seed(42)
        
        for bin_class in y_binned_resampled:
            stats = bin_stats[bin_class]
            synthetic_value = np.random.normal(stats['mean'], stats['std'] * 0.5)
            synthetic_value = np.clip(synthetic_value, stats['min'], stats['max'])
            y_resampled.append(synthetic_value)
        
        y_resampled = np.array(y_resampled)
        
        df_ampliado = pd.DataFrame(X_resampled, columns=X.columns)
        df_ampliado['target'] = y_resampled
        df_ampliado.to_csv(archivo_ampliado, index=False)
        
        print(f"Dataset ampliado con factor reducido: {factor_reducido}x")
        print(f"Nuevas dimensiones: {df_ampliado.shape}")
        print(f"Archivo guardado: {archivo_ampliado}")
        
        return df_ampliado


def crear_visualizacion_comparativa(df_original, df_ampliado):
    """Crea visualizaciones comparativas entre datasets original y ampliado."""
    
    plt.figure(figsize=(15, 10))
    
    # Distribución del target
    plt.subplot(2, 3, 1)
    plt.hist(df_original['target'], bins=30, alpha=0.7, label='Original', density=True)
    plt.hist(df_ampliado['target'], bins=30, alpha=0.7, label='SMOTE', density=True)
    plt.xlabel('Target')
    plt.ylabel('Densidad')
    plt.title('Distribución del Target')
    plt.legend()
    
    # Boxplot comparativo del target
    plt.subplot(2, 3, 2)
    data_box = [df_original['target'], df_ampliado['target']]
    plt.boxplot(data_box, labels=['Original', 'SMOTE'])
    plt.ylabel('Target')
    plt.title('Boxplot del Target')
    
    # Correlación entre features (muestra)
    plt.subplot(2, 3, 3)
    features_sample = df_original.columns[:5]  # Primeras 5 features
    corr_original = df_original[features_sample].corr()
    sns.heatmap(corr_original, annot=True, cmap='coolwarm', center=0, 
                square=True, cbar_kws={'shrink': 0.8})
    plt.title('Correlación Original')
    
    plt.subplot(2, 3, 4)
    corr_smote = df_ampliado[features_sample].corr()
    sns.heatmap(corr_smote, annot=True, cmap='coolwarm', center=0,
                square=True, cbar_kws={'shrink': 0.8})
    plt.title('Correlación SMOTE')
    
    # Scatter plot de dos features principales
    plt.subplot(2, 3, 5)
    plt.scatter(df_original.iloc[:, 0], df_original.iloc[:, 1], 
                alpha=0.6, label='Original', s=20)
    plt.scatter(df_ampliado.iloc[:, 0], df_ampliado.iloc[:, 1], 
                alpha=0.3, label='SMOTE', s=10)
    plt.xlabel(df_original.columns[0])
    plt.ylabel(df_original.columns[1])
    plt.title('Scatter Plot Features')
    plt.legend()
    
    # Estadísticas de tamaño
    plt.subplot(2, 3, 6)
    sizes = [len(df_original), len(df_ampliado)]
    labels = ['Original', 'SMOTE']
    colors = ['lightblue', 'lightcoral']
    plt.bar(labels, sizes, color=colors)
    plt.ylabel('Número de muestras')
    plt.title('Tamaño de datasets')
    
    # Añadir valores en las barras
    for i, v in enumerate(sizes):
        plt.text(i, v + max(sizes) * 0.01, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('comparacion_diabetes_smote.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nGráfico guardado como: comparacion_diabetes_smote.png")


if __name__ == "__main__":
    # Aplicar SMOTE con factor de aumento de 50x (de ~400 a ~20,000 muestras)
    df_ampliado = aplicar_smote_diabetes(
        archivo_original='diabetes.csv',
        archivo_ampliado='diabetes_smote.csv',
        factor_aumento=50
    )
    
    print("\n" + "=" * 70)
    print("SMOTE APLICADO EXITOSAMENTE")
    print("=" * 70)
    print("El dataset de diabetes ha sido ampliado usando SMOTE.")
    print("Ahora puedes usar 'diabetes_smote.csv' para entrenar con más datos.")
    print("Este dataset mantiene las características estadísticas del original")
    print("pero con muchas más muestras sintéticas realistas.")