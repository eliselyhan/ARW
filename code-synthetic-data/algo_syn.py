import numpy as np
from ARW import tournament_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb


# Training

def train_synthetic(X_train, B_arr_train, window_sizes_train):

    """ given 1D array of length B_1 + ... + B_T, and a list of m window sizes,
    compute the corresponding running averages

    Args:
        X_train (np.array): 1D array of length B_1 + ... + B_t
        B_arr_train (array): stores the sample size for each period
        window_sizes_train (list): window sizes

    Returns:
        models (list of np.arrays): len = m; each element is a 1D array
    """

    start_idcs = np.cumsum(B_arr_train) - B_arr_train
    models = [] 
    for k in window_sizes_train:
        start = start_idcs[max(len(B_arr_train) - k, 0)]
        models.append( np.mean( X_train[start:] ) )
        
    return models


# Selection - fixed window

def select_synthetic_fixed(X_val, B_arr_val, models, window_sizes_val):

    """ evaluate models on the validation data within given windows
    Args:
        X_val (np.array): 1D array of length B_1 + ... + B_t
        B_arr_val (array): stores the sample size for each period
        models (list): list of predictions by different models 
        window_sizes_val (list): list of window sizes for evaluation
    Returns:
        indices_selected (list): indices of selected models
        models_selected (list): selected models
    """

    start_idcs = np.cumsum(B_arr_val) - B_arr_val
    models_selected = []
    indices_selected = []
    for window_size in window_sizes_val:
        start = start_idcs[max(len(B_arr_val) - window_size, 0)]
        losses = np.array([ np.mean( (X_val[start:] - model) ** 2 ) for model in models ])
        r_hat = np.argmin(losses)
        indices_selected.append(r_hat)
        models_selected.append(models[r_hat])

    return indices_selected, models_selected


# selection - new method

def select_synthetic_ARW(X_val, B_arr_val, models, delta = 0.1, M = 10, seed = 2024):
    losses = [ (X_val - model) ** 2 for model in models ]
    idx_selected = tournament_selection(losses, B_arr_val, delta, M, seed)
    return idx_selected, models[idx_selected]

def train_random_forest(X_train, y_train, B_arr_train, window_sizes_train):

    start_idcs = np.cumsum(B_arr_train) - B_arr_train

    models = [] 

    for k in window_sizes_train:

        #take all data in the k recent months
        start = start_idcs[max(len(B_arr_train) - k, 0)]
        X_tk = X_train[start:,:]
        y_tk = y_train[start:]

        #train a random forest on this data
        rf = RandomForestRegressor(n_estimators=20, random_state=0)
        rf.fit(X_tk, y_tk)
        models.append(rf)
    
    return models

def select_model_fixed(X_val, y_val, B_arr_val, models, window_sizes_val):

    start_idcs = np.cumsum(B_arr_val) - B_arr_val

    models_selected = []
    indices_selected = []

    for k in window_sizes_val:

        #get the most recent k months' data
        start = start_idcs[max(len(B_arr_val) - k, 0)]
        X_tk = X_val[start:,:]
        y_tk = y_val[start:]

        #compute losses on X_tk, y_tk
        losses = []
        for model in models:
            y_pred = model.predict(X_tk)
            loss = np.mean((y_tk- y_pred)**2)
            losses.append(loss)
        
        #select the best models
        r_hat = np.argmin(losses)
        indices_selected.append(r_hat)
        models_selected.append(models[r_hat])

    return indices_selected, models_selected

def select_model_ARW(X_val, y_val, B_arr_val, models, delta = 0.1, M = 10, seed = 2024):
    losses = []
    for model in models:
        y_pred = model.predict(X_val)
        loss = (y_val- y_pred)**2
        losses.append(loss)
    idx_selected = tournament_selection(np.array(losses), B_arr_val, delta, M, seed)
    return idx_selected, models[idx_selected]

def train_linear_regression(X_train, y_train, B_arr_train, window_sizes_train):

    start_idcs = np.cumsum(B_arr_train) - B_arr_train

    models = [] 

    for k in window_sizes_train:

        #take all data in the k recent months
        start = start_idcs[max(len(B_arr_train) - k, 0)]
        X_tk = X_train[start:,:]
        y_tk = y_train[start:]

        #train a linear regression on this data
        xgb_reg  = xgb.XGBRegressor()
        xgb_reg.fit(X_tk, y_tk)
        models.append(xgb_reg)
    
    return models