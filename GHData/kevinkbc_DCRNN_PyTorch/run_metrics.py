import numpy as np
import lib.metrics

if __name__ == '__main__':
    filename = 'data/dcrnn_predictions.npz'
    savedPrediction = np.load(filename)
    mae, mape, rmse = lib.metrics.calculate_metrics(savedPrediction['prediction'], savedPrediction['truth'], 0.)
    print("MAE : {}".format(mae))
    print("MAPE : {}".format(mape))
    print("RMSE : {}".format(rmse))

