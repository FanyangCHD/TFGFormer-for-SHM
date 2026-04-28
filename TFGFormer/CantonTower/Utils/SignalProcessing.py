import numpy as np
import math

def calculate_mae(y_pred, y_true):
    y_pred = y_pred.data.cpu().numpy() 
    y_true = y_true.data.cpu().numpy() 
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def calculate_rmse(y_pred,y_true):
    y_pred = y_pred.data.cpu().numpy() 
    y_true = y_true.data.cpu().numpy() 
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    mse1 = math.sqrt(mse)
    return mse1

def calculate_RE(y_pred,y_true):
    y_pred = y_pred.data.cpu().numpy() 
    y_true = y_true.data.cpu().numpy() 
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    num1 = np.sum((y_true - y_pred) ** 2)
    num = np.sqrt(num1)
    den1 = np.sum(y_true ** 2)
    den = np.sqrt(den1)
    return num / den

def calculate_r2(y_pred, y_true):
    y_pred = y_pred.cpu().detach().numpy().flatten()
    y_true = y_true.cpu().detach().numpy().flatten()
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mean_true = np.mean(y_true)
    total_sum_squares = np.sum((y_true - mean_true) ** 2)
    residual_sum_squares = np.sum((y_true - y_pred) ** 2)
    if total_sum_squares == 0:
        if residual_sum_squares == 0:
            return 1.0  
        else:
            return 0.0  
    r_squared = 1 - (residual_sum_squares / total_sum_squares)
    return float(r_squared)



