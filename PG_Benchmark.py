import numpy as np
import pandas as pd
import time
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from evopt import EvolutionaryOptimizer
import warnings
warnings.filterwarnings('ignore')


class GPBenchmarkSuite:
    """Suite exhaustiva de benchmarks para Programación Genética."""
    
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Baseline
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        baseline = LinearRegression()
        baseline.fit(X_train_scaled, self.y_train)
        self.baseline_mae = mean_absolute_error(self.y_test, baseline.predict(X_test_scaled))
        self.baseline_r2 = r2_score(self.y_test, baseline.predict(X_test_scaled))
        
        print(f"Baseline - MAE: {self.baseline_mae:.4f}, R²: {self.baseline_r2:.4f}")
        
        self.results = []
        
        # Conjuntos de funciones para probar
        self.function_sets = {
            'basic': {'add': 2, 'sub': 2, 'mul': 2, 'div': 2},
            'extended': {'add': 2, 'sub': 2, 'mul': 2, 'div': 2, 'sqrt': 1, 'square': 1, 'abs': 1},
            'full': {'add': 2, 'sub': 2, 'mul': 2, 'div': 2, 'sqrt': 1, 'square': 1, 'log': 1, 'abs': 1, 'sin': 1, 'cos': 1},
            'math_heavy': {'mul': 2, 'div': 2, 'sqrt': 1, 'square': 1, 'log': 1, 'sin': 1, 'cos': 1},
            'arithmetic': {'add': 2, 'sub': 2, 'mul': 2, 'div': 2, 'square': 1, 'abs': 1}
        }
    
    def run_population_analysis(self):
        """Análisis exhaustivo del tamaño de población."""
        print(f"\n{'='*80}")
        print(f"BENCHMARK GP: ANÁLISIS DE POBLACIÓN")
        print(f"{'='*80}")
        
        base_config = {
            'maxtime': 900,  # 15 min por prueba
            'n_features_to_create': 8,
            'mutation_prob': 0.2,
            'crossover_prob': 0.8,
            'tournament_size': 5,
            'max_depth': 5,
            'elite_size': 0.1
        }
        
        population_sizes = [20, 30, 40, 50, 60, 80, 100, 120, 150, 200]
        
        for pop_size in population_sizes:
            config = base_config.copy()
            config['population_size'] = pop_size
            
            print(f"\nPoblación: {pop_size}")
            result = self._test_gp_configuration(config, f"Pop_{pop_size}")
            self.results.append(result)
            
            if 'error' not in result:
                print(f"  ✓ MAE: {result['mae']:.4f} | Mejora: {result['improvement']:.2f}%")
            else:
                print(f"  ✗ ERROR: {result['error']}")
    
    def run_features_analysis(self):
        """Análisis del número de features a crear."""
        print(f"\n{'='*80}")
        print(f"BENCHMARK GP: ANÁLISIS DE FEATURES")
        print(f"{'='*80}")
        
        base_config = {
            'maxtime': 900,
            'population_size': 80,
            'mutation_prob': 0.2,
            'crossover_prob': 0.8,
            'tournament_size': 5,
            'max_depth': 5,
            'elite_size': 0.1
        }
        
        n_features_list = [3, 5, 6, 8, 10, 12, 15, 18, 20, 25]
        
        for n_features in n_features_list:
            config = base_config.copy()
            config['n_features_to_create'] = n_features
            
            print(f"\nFeatures a crear: {n_features}")
            result = self._test_gp_configuration(config, f"Feat_{n_features}")
            self.results.append(result)
            
            if 'error' not in result:
                print(f"  ✓ MAE: {result['mae']:.4f} | Mejora: {result['improvement']:.2f}%")
            else:
                print(f"  ✗ ERROR: {result['error']}")
    
    def run_depth_analysis(self):
        """Análisis de profundidad máxima de árboles."""
        print(f"\n{'='*80}")
        print(f"BENCHMARK GP: ANÁLISIS DE PROFUNDIDAD")
        print(f"{'='*80}")
        
        base_config = {
            'maxtime': 900,
            'population_size': 80,
            'n_features_to_create': 8,
            'mutation_prob': 0.2,
            'crossover_prob': 0.8,
            'tournament_size': 5,
            'elite_size': 0.1
        }
        
        max_depths = [2, 3, 4, 5, 6, 7, 8, 10]
        
        for depth in max_depths:
            config = base_config.copy()
            config['max_depth'] = depth
            
            print(f"\nProfundidad máxima: {depth}")
            result = self._test_gp_configuration(config, f"Depth_{depth}")
            self.results.append(result)
            
            if 'error' not in result:
                print(f"  ✓ MAE: {result['mae']:.4f} | Mejora: {result['improvement']:.2f}%")
                print(f"    Mejor árbol: {result.get('best_tree', 'N/A')}")
            else:
                print(f"  ✗ ERROR: {result['error']}")
    
    def run_function_set_analysis(self):
        """Análisis de diferentes conjuntos de funciones."""
        print(f"\n{'='*80}")
        print(f"BENCHMARK GP: ANÁLISIS DE CONJUNTOS DE FUNCIONES")
        print(f"{'='*80}")
        
        base_config = {
            'maxtime': 1200,  # Más tiempo para funciones complejas
            'population_size': 80,
            'n_features_to_create': 8,
            'mutation_prob': 0.2,
            'crossover_prob': 0.8,
            'tournament_size': 5,
            'max_depth': 5,
            'elite_size': 0.1
        }
        
        for func_name, func_set in self.function_sets.items():
            config = base_config.copy()
            config['function_set'] = func_set
            
            print(f"\nConjunto de funciones: {func_name}")
            print(f"  Funciones: {list(func_set.keys())}")
            result = self._test_gp_configuration(config, f"FuncSet_{func_name}")
            self.results.append(result)
            
            if 'error' not in result:
                print(f"  ✓ MAE: {result['mae']:.4f} | Mejora: {result['improvement']:.2f}%")
            else:
                print(f"  ✗ ERROR: {result['error']}")
    
    def run_genetic_operators_analysis(self):
        """Análisis de operadores genéticos."""
        print(f"\n{'='*80}")
        print(f"BENCHMARK GP: ANÁLISIS DE OPERADORES GENÉTICOS")
        print(f"{'='*80}")
        
        base_config = {
            'maxtime': 900,
            'population_size': 80,
            'n_features_to_create': 8,
            'max_depth': 5,
            'tournament_size': 5,
            'elite_size': 0.1
        }
        
        # Combinaciones de mutación y cruce
        operator_combinations = [
            (0.1, 0.7), (0.1, 0.8), (0.1, 0.9),
            (0.15, 0.7), (0.15, 0.8), (0.15, 0.9),
            (0.2, 0.7), (0.2, 0.8), (0.2, 0.9),
            (0.25, 0.7), (0.25, 0.8), (0.25, 0.9),
            (0.3, 0.7), (0.3, 0.8), (0.3, 0.9),
            (0.35, 0.8), (0.4, 0.8), (0.5, 0.8)
        ]
        
        for mut_prob, cross_prob in operator_combinations:
            config = base_config.copy()
            config['mutation_prob'] = mut_prob
            config['crossover_prob'] = cross_prob
            
            print(f"\nMutación: {mut_prob} | Cruce: {cross_prob}")
            result = self._test_gp_configuration(config, f"Op_{mut_prob}_{cross_prob}")
            self.results.append(result)
            
            if 'error' not in result:
                print(f"  ✓ MAE: {result['mae']:.4f} | Mejora: {result['improvement']:.2f}%")
            else:
                print(f"  ✗ ERROR: {result['error']}")
    
    def run_tournament_elite_analysis(self):
        """Análisis de selección por torneo y elitismo."""
        print(f"\n{'='*80}")
        print(f"BENCHMARK GP: ANÁLISIS DE SELECCIÓN Y ELITISMO")
        print(f"{'='*80}")
        
        base_config = {
            'maxtime': 900,
            'population_size': 80,
            'n_features_to_create': 8,
            'mutation_prob': 0.2,
            'crossover_prob': 0.8,
            'max_depth': 5
        }
        
        # Combinaciones de torneo y elite
        selection_combinations = [
            (3, 0.05), (3, 0.1), (3, 0.15),
            (5, 0.05), (5, 0.1), (5, 0.15), (5, 0.2),
            (7, 0.1), (7, 0.15), (7, 0.2),
            (10, 0.1), (10, 0.15), (10, 0.2), (10, 0.25)
        ]
        
        for tournament_size, elite_size in selection_combinations:
            config = base_config.copy()
            config['tournament_size'] = tournament_size
            config['elite_size'] = elite_size
            
            print(f"\nTorneo: {tournament_size} | Elite: {elite_size*100:.0f}%")
            result = self._test_gp_configuration(config, f"Sel_{tournament_size}_{elite_size}")
            self.results.append(result)
            
            if 'error' not in result:
                print(f"  ✓ MAE: {result['mae']:.4f} | Mejora: {result['improvement']:.2f}%")
            else:
                print(f"  ✗ ERROR: {result['error']}")
    
    def run_time_analysis(self):
        """Análisis del tiempo de ejecución."""
        print(f"\n{'='*80}")
        print(f"BENCHMARK GP: ANÁLISIS DE TIEMPO")
        print(f"{'='*80}")
        
        base_config = {
            'population_size': 80,
            'n_features_to_create': 8,
            'mutation_prob': 0.2,
            'crossover_prob': 0.8,
            'tournament_size': 5,
            'max_depth': 5,
            'elite_size': 0.1
        }
        
        times = [300, 600, 900, 1200, 1800, 2400, 3600]  # 5min a 1h
        
        for maxtime in times:
            config = base_config.copy()
            config['maxtime'] = maxtime
            
            print(f"\nTiempo: {maxtime//60} min")
            result = self._test_gp_configuration(config, f"Time_{maxtime//60}min")
            self.results.append(result)
            
            if 'error' not in result:
                print(f"  ✓ MAE: {result['mae']:.4f} | Mejora: {result['improvement']:.2f}% | Gens: {result['generations']}")
            else:
                print(f"  ✗ ERROR: {result['error']}")
    
    def run_comprehensive_grid_search(self):
        """Búsqueda en grid de las mejores configuraciones encontradas."""
        print(f"\n{'='*80}")
        print(f"BENCHMARK GP: BÚSQUEDA COMPREHENSIVA EN GRID")
        print(f"{'='*80}")
        
        # Configuraciones prometedoras basadas en análisis previos
        promising_configs = [
            # Configuración balanceada
            {'maxtime': 1200, 'population_size': 80, 'n_features_to_create': 8, 
             'mutation_prob': 0.2, 'crossover_prob': 0.8, 'tournament_size': 5, 
             'max_depth': 5, 'elite_size': 0.1, 'function_set': self.function_sets['extended']},
            
            # Configuración intensiva
            {'maxtime': 1800, 'population_size': 100, 'n_features_to_create': 10, 
             'mutation_prob': 0.25, 'crossover_prob': 0.8, 'tournament_size': 7, 
             'max_depth': 6, 'elite_size': 0.15, 'function_set': self.function_sets['full']},
            
            # Configuración rápida
            {'maxtime': 600, 'population_size': 60, 'n_features_to_create': 6, 
             'mutation_prob': 0.15, 'crossover_prob': 0.9, 'tournament_size': 3, 
             'max_depth': 4, 'elite_size': 0.1, 'function_set': self.function_sets['basic']},
            
            # Configuración exploratoria
            {'maxtime': 1500, 'population_size': 120, 'n_features_to_create': 12, 
             'mutation_prob': 0.3, 'crossover_prob': 0.7, 'tournament_size': 5, 
             'max_depth': 7, 'elite_size': 0.2, 'function_set': self.function_sets['math_heavy']},
            
            # Configuración conservadora
            {'maxtime': 900, 'population_size': 50, 'n_features_to_create': 5, 
             'mutation_prob': 0.1, 'crossover_prob': 0.9, 'tournament_size': 3, 
             'max_depth': 3, 'elite_size': 0.05, 'function_set': self.function_sets['arithmetic']},
        ]
        
        for i, config in enumerate(promising_configs, 1):
            print(f"\n[{i}/{len(promising_configs)}] Configuración prometedora:")
            for k, v in config.items():
                if k != 'function_set':
                    print(f"  {k}: {v}")
                else:
                    print(f"  function_set: {list(v.keys())}")
            
            result = self._test_gp_configuration(config, f"Grid_{i}")
            self.results.append(result)
            
            if 'error' not in result:
                print(f"  ✓ MAE: {result['mae']:.4f} | Mejora: {result['improvement']:.2f}%")
            else:
                print(f"  ✗ ERROR: {result['error']}")
    
    def _test_gp_configuration(self, config, name):
        """Prueba una configuración específica de GP."""
        start_time = time.time()
        
        try:
            # Crear optimizador con función set personalizada si se especifica
            if 'function_set' in config:
                # Crear clase temporal con función set personalizada
                class CustomGPOptimizer(EvolutionaryOptimizer):
                    def __init__(self, **kwargs):
                        function_set = kwargs.pop('function_set', None)
                        super().__init__(**kwargs)
                        if function_set:
                            self.functions = function_set
                
                optimizer = CustomGPOptimizer(**config)
            else:
                optimizer = EvolutionaryOptimizer(**config)
            
            optimizer.fit(self.X_train, self.y_train)
            y_pred = optimizer.predict(self.X_test)
            
            # Verificar predicciones válidas
            if y_pred is None or np.any(np.isnan(y_pred)) or np.any(np.abs(y_pred) > 1e6):
                return {
                    'name': name, 'config': config, 'mae': 1e6, 'r2': -1e6,
                    'improvement': -1e6, 'time': time.time() - start_time,
                    'generations': 0, 'best_fitness': 1e6, 'error': 'Invalid predictions'
                }
            
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            if np.isnan(mae) or np.isnan(r2) or mae > 1e6:
                return {
                    'name': name, 'config': config, 'mae': 1e6, 'r2': -1e6,
                    'improvement': -1e6, 'time': time.time() - start_time,
                    'generations': len(optimizer.fitness_history_), 'best_fitness': 1e6,
                    'error': 'Invalid metrics'
                }
            
            improvement = ((self.baseline_mae - mae) / self.baseline_mae) * 100
            
            result = {
                'name': name,
                'config': config,
                'mae': mae,
                'r2': r2,
                'improvement': improvement,
                'time': time.time() - start_time,
                'generations': len(optimizer.fitness_history_),
                'best_fitness': optimizer.best_fitness_
            }
            
            # Añadir mejor árbol si existe
            if optimizer.best_trees_:
                result['best_tree'] = optimizer.best_trees_[0].to_string()
            
            return result
            
        except Exception as e:
            return {
                'name': name, 'config': config, 'mae': 1e6, 'r2': -1e6,
                'improvement': -1e6, 'time': time.time() - start_time,
                'generations': 0, 'best_fitness': 1e6, 'error': str(e)
            }
    
    def generate_comprehensive_report(self):
        """Genera reporte exhaustivo de todos los resultados."""
        print(f"\n{'='*100}")
        print(f"REPORTE EXHAUSTIVO DE PROGRAMACIÓN GENÉTICA")
        print(f"{'='*100}")
        
        # Filtrar resultados válidos
        valid_results = [r for r in self.results if 'error' not in r]
        error_results = [r for r in self.results if 'error' in r]
        
        print(f"\nRESUMEN GENERAL:")
        print(f"  Total configuraciones probadas: {len(self.results)}")
        print(f"  Configuraciones exitosas: {len(valid_results)}")
        print(f"  Configuraciones fallidas: {len(error_results)}")
        
        if not valid_results:
            print("\n❌ NO SE ENCONTRARON CONFIGURACIONES VÁLIDAS")
            return None
        
        # Ordenar por mejora
        sorted_results = sorted(valid_results, key=lambda x: x['improvement'], reverse=True)
        
        print(f"\nTOP 15 MEJORES CONFIGURACIONES:")
        print(f"{'Rank':<4} {'Nombre':<20} {'MAE':<8} {'R²':<8} {'Mejora%':<8} {'Tiempo':<8} {'Gens':<6}")
        print(f"{'-'*80}")
        
        for i, result in enumerate(sorted_results[:15], 1):
            print(f"{i:<4} ✓ {result['name']:<18} {result['mae']:<8.4f} {result['r2']:<8.4f} " +
                  f"{result['improvement']:<8.2f} {result['time']/60:<8.1f} {result['generations']:<6}")
        
        # Mejor configuración
        best = sorted_results[0]
        print(f"\n{'='*60}")
        print(f"🏆 MEJOR CONFIGURACIÓN ENCONTRADA:")
        print(f"{'='*60}")
        print(f"Nombre: {best['name']}")
        print(f"MAE: {best['mae']:.4f} (Baseline: {self.baseline_mae:.4f})")
        print(f"R²: {best['r2']:.4f} (Baseline: {self.baseline_r2:.4f})")
        print(f"Mejora: {best['improvement']:.2f}%")
        print(f"Tiempo: {best['time']/60:.1f} minutos")
        print(f"Generaciones: {best['generations']}")
        
        if 'best_tree' in best:
            print(f"Mejor árbol: {best['best_tree']}")
        
        print(f"\nParámetros óptimos:")
        for k, v in best['config'].items():
            if k != 'function_set':
                print(f"  {k}: {v}")
            else:
                print(f"  function_set: {list(v.keys())}")
        
        # Análisis por categorías
        self._analyze_categories(valid_results)
        
        return best
    
    def _analyze_categories(self, results):
        """Análisis detallado por categorías."""
        print(f"\n{'='*60}")
        print(f"ANÁLISIS POR CATEGORÍAS")
        print(f"{'='*60}")
        
        # Análisis de población
        pop_results = [r for r in results if 'Pop_' in r['name']]
        if pop_results:
            best_pop = max(pop_results, key=lambda x: x['improvement'])
            print(f"\n🔸 Mejor población: {best_pop['config']['population_size']} " +
                  f"(Mejora: {best_pop['improvement']:.2f}%)")
        
        # Análisis de features
        feat_results = [r for r in results if 'Feat_' in r['name']]
        if feat_results:
            best_feat = max(feat_results, key=lambda x: x['improvement'])
            print(f"🔸 Mejores features: {best_feat['config']['n_features_to_create']} " +
                  f"(Mejora: {best_feat['improvement']:.2f}%)")
        
        # Análisis de profundidad
        depth_results = [r for r in results if 'Depth_' in r['name']]
        if depth_results:
            best_depth = max(depth_results, key=lambda x: x['improvement'])
            print(f"🔸 Mejor profundidad: {best_depth['config']['max_depth']} " +
                  f"(Mejora: {best_depth['improvement']:.2f}%)")
        
        # Análisis de funciones
        func_results = [r for r in results if 'FuncSet_' in r['name']]
        if func_results:
            best_func = max(func_results, key=lambda x: x['improvement'])
            func_name = best_func['name'].replace('FuncSet_', '')
            print(f"🔸 Mejor conjunto de funciones: {func_name} " +
                  f"(Mejora: {best_func['improvement']:.2f}%)")
        
        # Análisis de tiempo
        time_results = [r for r in results if 'Time_' in r['name']]
        if time_results:
            best_time = max(time_results, key=lambda x: x['improvement'])
            print(f"🔸 Mejor tiempo: {best_time['config']['maxtime']//60} min " +
                  f"(Mejora: {best_time['improvement']:.2f}%)")


def run_exhaustive_gp_benchmark():
    """Ejecuta benchmark exhaustivo de Programación Genética."""
    
    # Cargar datos
    df = pd.read_csv('Comp_Ev/diabetes.csv')
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    print(f"🧬 BENCHMARK EXHAUSTIVO DE PROGRAMACIÓN GENÉTICA")
    print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} features")
    print(f"Tiempo estimado: 4-6 horas")
    
    # Crear suite de benchmark
    benchmark = GPBenchmarkSuite(X, y)
    
    # Ejecutar todos los análisis
    print(f"\n🚀 Iniciando análisis exhaustivo...")
    
    benchmark.run_population_analysis()
    benchmark.run_features_analysis()
    benchmark.run_depth_analysis()
    benchmark.run_function_set_analysis()
    benchmark.run_genetic_operators_analysis()
    benchmark.run_tournament_elite_analysis()
    benchmark.run_time_analysis()
    benchmark.run_comprehensive_grid_search()
    
    # Generar reporte final
    best_config = benchmark.generate_comprehensive_report()
    
    return best_config


if __name__ == "__main__":
    print("🧬 Iniciando benchmark exhaustivo de Programación Genética...")
    print("⚠️  ADVERTENCIA: Este proceso puede tardar 4-6 horas")
    
    response = input("¿Continuar? (y/n): ")
    if response.lower() == 'y':
        best_config = run_exhaustive_gp_benchmark()
        print(f"\n🎉 Benchmark completado. Mejor configuración encontrada.")
    else:
        print("Benchmark cancelado.")