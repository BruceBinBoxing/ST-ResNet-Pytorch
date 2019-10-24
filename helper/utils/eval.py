
from __future__ import print_function
import numpy as np
# np.random.seed(1337)  # for reproducibility


def rmse(Y_true, Y_pred):
    # https://www.kaggle.com/wiki/RootMeanSquaredError
    from sklearn.metrics import mean_squared_error
    print('shape:', Y_true.shape, Y_pred.shape)
    print("===RMSE===")
    # in
    RMSE = mean_squared_error(Y_true[:, 0].flatten(), Y_pred[:, 0].flatten())**0.5
    print('inflow: ', RMSE)
    # out
    if Y_true.shape[1] > 1:
        RMSE = mean_squared_error(Y_true[:, 1].flatten(), Y_pred[:, 1].flatten())**0.5
        print('outflow: ', RMSE)
    # new
    if Y_true.shape[1] > 2:
        RMSE = mean_squared_error(Y_true[:, 2].flatten(), Y_pred[:, 2].flatten())**0.5
        print('newflow: ', RMSE)
    # end
    if Y_true.shape[1] > 3:
        RMSE = mean_squared_error(Y_true[:, 3].flatten(), Y_pred[:, 3].flatten())**0.5
        print('endflow: ', RMSE)

    RMSE = mean_squared_error(Y_true.flatten(), Y_pred.flatten())**0.5
    print("total rmse: ", RMSE)
    print("===RMSE===")
    return RMSE


def mean_absolute_percentage_error(y_true, y_pred):
    idx = np.nonzero(y_true)
    return np.mean(np.abs((y_true[idx] - y_pred[idx]) / y_true[idx])) * 100


def mape(Y_true, Y_pred):
    print("===MAPE===")
    # in
    MAPE = mean_absolute_percentage_error(Y_true[:, 0].flatten(), Y_pred[:, 0].flatten())
    print("inflow: ", MAPE)
    # out
    MAPE = mean_absolute_percentage_error(Y_true[:, 1].flatten(), Y_pred[:, 1].flatten())
    print("outflow: ", MAPE)
    # new
    MAPE = mean_absolute_percentage_error(Y_true[:, 2].flatten(), Y_pred[:, 2].flatten())
    print("newflow: ", MAPE)
    # end
    MAPE = mean_absolute_percentage_error(Y_true[:, 3].flatten(), Y_pred[:, 3].flatten())
    print("endflow: ", MAPE)
    MAPE = mean_absolute_percentage_error(Y_true.flatten(), Y_pred.flatten())
    print("total mape: ", MAPE)
    print("===MAPE===")
    return MAPE
