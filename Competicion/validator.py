import importlib
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

TRAIN_FILE = 'california_train.csv'
TEST_FILE = 'california_test.csv'

def benchmark (Xt, yt, Xv, yv):
    """
    Tests the performance of a dataset over a bunch of ML models.
    NOTICE: this benchmark is simplified, the final one will have more models.

    @param Xt: feature matrix of training set.
    @param yt: output vector of training set.
    @param Xv: feature matrix of test set.
    @param yv: output vector of test set.
    @return average error.
    """
    lm = LinearRegression()
    #lm = RandomForestRegressor(n_jobs=-1, random_state=42)
    lm.fit(Xt, yt)
    yp = lm.predict(Xv)
    e = mean_squared_error(yv, yp)
    return e

    
if __name__ == '__main__':

    # Load input files.
    T = pd.read_csv(TRAIN_FILE)
    V = pd.read_csv(TEST_FILE)

    # Extract X and y.
    Xt = T.iloc[:,:-1].values
    yt = T.iloc[:,-1].values.reshape(-1, 1)
    Xv = V.iloc[:,:-1].values 
    yv = V.iloc[:,-1].values.reshape(-1, 1)

    # Compute baseline.
    baseline = benchmark(Xt, yt, Xv, yv)
    
    # Iterate through models.
    for folder in filter(lambda f : os.path.isdir(f), os.listdir('.')):
        try:
            evomod = importlib.import_module(f'{folder}.evopt', package=None)
            evo = evomod.EvolutionaryOptimizer(maxtime=3600)  # This will be 3600 seconds!
            evo.fit(Xt, yt)
            Xt_ = evo.transform(Xt)
            Xv_ = evo.transform(Xv)
            error = benchmark(Xt_, yt, Xv_, yv)
            print(f'{folder}: baseline={baseline:.4f}, optimized={error:.4f}, metric={(1 - error / baseline):.4f}')
        except Exception as e:
            print(f'Ha ocurrido un error con {folder}: {e}')