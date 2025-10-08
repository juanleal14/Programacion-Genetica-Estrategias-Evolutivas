# Programacion-Genetica-Estrategias-Evolutivas
# Resumen de `evopt2.py`

Este documento resume la estructura y el flujo principal del módulo `evopt2.py`, que implementa una cadena de Programación Genética (PG) combinada con selección evolutiva de variables para problemas de regresión.

## 1. Representación de árboles GP

* `GPNode` define la unidad básica de un árbol de programación genética.
  * Cada nodo guarda un `value`, una lista de `children` y un `node_type` que distingue entre terminales (índices de variables o constantes) y funciones.【F:evopt2.py†L16-L77】
  * El método `evaluate` aplica la operación correspondiente sobre una matriz de entrada `X`, controlando posibles problemas numéricos (divisiones por cero, raíces negativas, logs) mediante desplazamientos y `np.clip` cuando es necesario.【F:evopt2.py†L24-L52】
  * Otros métodos proveen utilidades de clonación (`copy`), cálculo de tamaño y profundidad, y la representación textual del árbol (`to_string`) útil para depuración y reporte.【F:evopt2.py†L54-L77】

## 2. `EvolutionaryOptimizer`

`EvolutionaryOptimizer` hereda de los mixins de *scikit-learn* para integrarse con pipelines. Su objetivo es descubrir nuevas variables derivadas que mejoren el desempeño de un modelo lineal.

* **Inicialización**: define hiperparámetros de la evolución (tamaño poblacional, probabilidades de mutación/cruce, profundidad máxima) y el catálogo de funciones disponibles para construir árboles.【F:evopt2.py†L81-L108】
* **Entrenamiento (`fit`)**:
  * Escala las variables con `RobustScaler`, guarda el conteo de atributos originales y crea la población inicial de individuos, donde cada individuo es una lista de árboles GP.【F:evopt2.py†L111-L156】
  * Cada generación evalúa a los individuos midiendo el MAE negativo promedio en validación cruzada de 3 pliegues sobre un modelo `LinearRegression` entrenado con las variables originales más las derivadas por los árboles. Se añade una penalización proporcional a la complejidad total (número de nodos).【F:evopt2.py†L200-L241】
  * El mejor individuo encontrado se guarda en `best_trees_` y se actualiza `model_` con un ajuste final sobre todo el conjunto si mejora el fitness.【F:evopt2.py†L223-L241】
  * Se utilizan mecanismos clásicos de evolución: elitismo, selección por torneo, cruce y mutación tanto a nivel de lista de árboles como dentro de cada árbol individual.【F:evopt2.py†L162-L215】【F:evopt2.py†L248-L339】
  * Existe una estrategia de reinicio (“cambio de rama”) cuando el algoritmo se estanca 200 generaciones, reconstruyendo la mayoría de la población con árboles más profundos para incrementar la diversidad.【F:evopt2.py†L138-L189】
* **Transformación (`transform`)**: reutiliza el `RobustScaler` ajustado, evalúa cada árbol óptimo sobre los datos escalados y concatena las nuevas variables a la matriz original (con limpieza numérica).【F:evopt2.py†L173-L194】
* **Predicción (`predict`)**: transforma y predice con la regresión lineal guardada en `model_`.【F:evopt2.py†L196-L198】

## 3. Generación y operadores evolutivos

Funciones privadas auxiliares encapsulan la lógica de generación de árboles, evaluación y operadores genéticos:

* `_create_random_tree` produce un individuo completo con `n_features_to_create` árboles y `_generate_tree` construye árboles recursivamente mezclando terminales (variables originales o constantes) y funciones del catálogo.【F:evopt2.py†L200-L227】
* `_evaluate_individual` monta el conjunto de variables derivadas, controla valores extremos, evalúa el MAE con validación cruzada y aplica penalización por complejidad.【F:evopt2.py†L228-L272】
* `_tournament_selection`, `_crossover_trees`, `_crossover_single_tree`, `_mutate_trees` y `_mutate_single_tree` implementan los operadores evolutivos sobre los conjuntos de árboles y sobre los nodos individuales.【F:evopt2.py†L274-L339】

## 4. Selección evolutiva de características

La segunda parte del archivo implementa un algoritmo evolutivo clásico (representación booleana) para seleccionar un subconjunto de variables una vez que la PG ha generado nuevas características.

* `evolutionary_feature_selection` crea una población inicial con al menos 3 variables activas, evalúa el MAE usando `LinearRegression` y utiliza elitismo, torneos, cruce uniforme y mutación de bits con restricciones para mantener al menos dos variables activas.【F:evopt2.py†L341-L428】
* `evaluate_feature_subset`, `tournament_selection_fs`, `crossover_fs` y `mutate_fs` contienen la lógica de evaluación y operadores para esta fase.【F:evopt2.py†L430-L504】

## 5. Script principal

El bloque `if __name__ == "__main__"` muestra un flujo completo de uso con el conjunto `diabetes.csv` (o `california.csv`). Calcula un baseline con regresión lineal, ejecuta la PG para crear nuevas variables, mide la mejora y finalmente aplica la selección evolutiva para refinar el conjunto de variables finales.【F:evopt2.py†L506-L642】

Este script imprime métricas de desempeño, los árboles generados y las variables seleccionadas, permitiendo analizar el impacto de cada fase del proceso evolutivo.