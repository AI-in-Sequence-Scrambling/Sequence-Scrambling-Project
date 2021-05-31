import numpy as np
import sklearn.metrics
import pandas as pd

from helper_functions import SQA_preprocessing as SQA_prepro


# Method to provide binary classification metrics
def binary_classification_metrics (prediction, y_test, pred_certainty):
    
    # Interpret the prediction as binary labels
    y_pred = (prediction > pred_certainty)
    
    # Calculate F-Scores, precision and recall
    f1_score = sklearn.metrics.f1_score(y_test, y_pred, average=None)
    print('F1-Scores = ' + str(f1_score))
    f2_score = sklearn.metrics.fbeta_score(y_test, y_pred, average=None, beta=2)
    print('F2-Scores = ' + str(f2_score))
    precision_score = sklearn.metrics.precision_score(y_test, y_pred, average=None)
    print('Precision = ' + str(precision_score))
    recall_score = sklearn.metrics.recall_score(y_test, y_pred, average=None)
    print('Recall = ' + str(recall_score))
    
    # Calculate confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    
    return f1_score, f2_score, precision_score, recall_score, cm



# Method to provide regression performance metrics   
def regression_metrics (prediction, y_test, X_test):
    
    # Call methods to provide the individual measures
    res_mse = mse(y_test, prediction)
    res_rmse = rmse(y_test, prediction)
    res_mae = mae(y_test, prediction)
    res_mape = mape(y_test, prediction)
    res_rae = rae(y_test, prediction)
    res_r_squared = r_squared(y_test, prediction)
    res_adj_r_squared = adj_r_squared(X_test, y_test, prediction)
    res_median_abs_error = median_abs_error(y_test, prediction)

    return res_mse, res_rmse, res_mae, res_mape, res_rae, res_r_squared, res_adj_r_squared, res_median_abs_error
    
    
    
# Methods to specify calculation of regression metrics    
def mse(actual, predicted):
    return np.mean(np.square(actual-predicted))


def rmse(actual, predicted):
    return np.sqrt(np.mean(np.square(actual-predicted)))
    
    
def mae(actual, predicted):
    return np.mean(np.abs(actual-predicted))
    
    
def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100
    
    
def rae(actual, predicted):
    numerator = np.sum(np.abs(predicted - actual))
    denominator = np.sum(np.abs(np.mean(actual) - actual))
    return numerator / denominator
     
     
def r_squared(actual, predicted):
    sse = np.sum(np.square(actual-predicted))
    sst = np.sum(np.square(actual-np.mean(actual)))
    return 1 - (sse/sst)
    
    
def adj_r_squared(X, actual, predicted): #X is your training dataset
    r_squ = r_squared(actual, predicted)
    numerator = 1 - (1 - r_squ) * len(actual)-1
    denominator = len(actual) - X.shape[1] #X.shape[1] will give number of independent variables
    return numerator/denominator
    
    
def median_abs_error(actual, predicted):
    return np.sum(np.median(np.abs(actual - predicted)))
   