import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import collections
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from ARW import tournament_selection

# Data:
# https://www.dubaipulse.gov.ae/data/dld-transactions/dld_transactions-open


# preprocessing
def preprocess(df, alpha = 0.05, encoding = 'label'):
    # encoding: 'label' for label encoding (1, 2, 3, ...), 'onehot' for one-hot encoding

    # select transactions of flats
    set_rooms = set(['Studio', '1 B/R', '2 B/R', '3 B/R', '4 B/R'])
    df = df[ (df['procedure_name_en'] == 'Sell') &\
            (df['property_sub_type_en'] == 'Flat')&\
            df['rooms_en'].isin(set_rooms) ]


    # select data between given years
    year_start, year_end = 2008, 2023
    tmp_years = np.array([int(x[6:10]) for x in df['instance_date']])
    df = df.iloc[(tmp_years >= year_start) & (tmp_years <= year_end)]
    df['instance_date'] = np.array([x[6:10]+x[3:5] for x in df['instance_date']])
    df = df.sort_values('instance_date')



    # select columns and drop missing values
    list_columns = ['instance_date', 'area_name_en', 'rooms_en',\
                    'has_parking', 'procedure_area', 'actual_worth']
    df = df[list_columns].dropna()


    # number of bedrooms
    dict_rooms = dict()
    for (i, r) in enumerate(set_rooms):
        dict_rooms[r] = i
    df.replace({'rooms_en': dict_rooms}, inplace = True)

    # log transform of housing price
    df['actual_worth'] = np.log(df['actual_worth'])

    # remove outliers
    tmp1_low = np.quantile(df['actual_worth'], alpha / 2)
    tmp1_high = np.quantile(df['actual_worth'], 1 - alpha / 2)
    tmp2_low = np.quantile(df['procedure_area'], alpha / 2)
    tmp2_high = np.quantile(df['procedure_area'], 1 - alpha / 2)

    tmp = np.mean((df['actual_worth'] >= tmp1_low) & (df['actual_worth'] <= tmp1_high) &\
            (df['procedure_area'] >= tmp2_low) & (df['procedure_area'] <= tmp2_high))
    #print('Fraction of remaining data:', tmp)


    df = df[ (df['actual_worth'] >= tmp1_low) & (df['actual_worth'] <= tmp1_high) &\
            (df['procedure_area'] >= tmp2_low) & (df['procedure_area'] <= tmp2_high)  ]

    # encoding
    if encoding == 'label':
        df['area_name_en'] = LabelEncoder().fit_transform(df['area_name_en'])
    if encoding == 'onehot':
        df = pd.get_dummies(df, columns=['area_name_en', ])
        df.replace({False: 0, True: 1}, inplace=True)

    return df



# data splitting
def splitting(df, prop, seed = 2024):
    months_counter = collections.Counter( df['instance_date'] )

    B_arr = []
    B_arr_train = []
    B_arr_val = []
    B_arr_test = []
    for key in months_counter:
        n = months_counter[key]
        n_train = int(n * prop[0])
        n_val = int(n * prop[1])   

        B_arr.append(n)
        B_arr_train.append(n_train)
        B_arr_val.append(n_val)
        B_arr_test.append(n - n_train - n_val)
        
    B_arr = np.array(B_arr)
    B_arr_train = np.array(B_arr_train)
    B_arr_val = np.array(B_arr_val)
    B_arr_test = np.array(B_arr_test)

    idx_train, idx_val, idx_test = sample_idx(B_arr, B_arr_train, B_arr_val, B_arr_test, seed)

    df_train = df.iloc[idx_train]
    df_val = df.iloc[idx_val]
    df_test = df.iloc[idx_test]

    X_train = df_train.drop(['instance_date', 'actual_worth'], axis=1)
    X_val = df_val.drop(['instance_date', 'actual_worth'], axis=1)
    X_test = df_test.drop(['instance_date', 'actual_worth'], axis=1)

    y_train = df_train['actual_worth']
    y_val = df_val['actual_worth']
    y_test = df_test['actual_worth']

    data_train = [X_train, y_train, B_arr_train]
    data_val = [X_val, y_val, B_arr_val]
    data_test = [X_test, y_test, B_arr_test]

    return [data_train, data_val, data_test]



# auxiliary function for data splitting
def sample_idx(B_arr, B_arr_train, B_arr_val, B_arr_test, seed):
    #get the end indices for each period
    end_idcs = np.cumsum(B_arr)
    end_idcs_train = np.cumsum(B_arr_train)
    end_idcs_val = np.cumsum(B_arr_val)
    end_idcs_test = np.cumsum(B_arr_test)

    
    idx_train = np.zeros(sum(B_arr_train))
    idx_val = np.zeros(sum(B_arr_val))    
    idx_test = np.zeros(sum(B_arr_test))

    np.random.seed(seed)
    #in each period, split data into train, validation and test
    for t in range(len(B_arr)):
        end = end_idcs[t] 
        start = end - B_arr[t]

        #randomly split indices into train, assess and test
        idx_t = np.arange(start, end)
        
        #partition the indices into train, assess and test
        idx_train_t = np.random.choice(idx_t, B_arr_train[t], replace=False)
        idx_val_t = np.random.choice(np.setdiff1d(idx_t, idx_train_t), B_arr_val[t], replace=False)
        idx_test_t = np.setdiff1d(idx_t, np.concatenate((idx_train_t, idx_val_t)))

        #get the corresponding data
        idx_train[(end_idcs_train[t] - B_arr_train[t]) : end_idcs_train[t]] = idx_train_t
        idx_val[(end_idcs_val[t] - B_arr_val[t]) : end_idcs_val[t]] = idx_val_t
        idx_test[(end_idcs_test[t] - B_arr_test[t]) : end_idcs_test[t]] = idx_test_t

    return idx_train, idx_val, idx_test



# Random forest and XGBoost

def test(data, t, window_sizes_train, window_sizes_val, delta = 0.1, M = 0, seed = 2024):
    [data_train, data_val, data_test] = data
    [X_train, y_train, B_arr_train] = data_train
    [X_val, y_val, B_arr_val] = data_val
    [X_test, y_test, B_arr_test] = data_test

    rf = RandomForestRegressor(random_state = 0)
    xgb = XGBRegressor(random_state = 0)
    scaler = StandardScaler()

    end_train = np.sum(B_arr_train[0:(t+1)])
    end_val = np.sum(B_arr_val[0:(t+1)])
    end_test = np.sum(B_arr_test[0:(t+1)])
    
    m = len(window_sizes_train)
    K = len(window_sizes_val)
    MSE_val_RF = np.zeros((K, m))
    MSE_test_RF = np.zeros(m)
    MSE_val_XGB = np.zeros((K, m))
    MSE_test_XGB = np.zeros(m)
    
    start_test = end_test - B_arr_test[t]
    losses_RF = []
    losses_XGB = []

    for i in range(m):
        # training
        r = window_sizes_train[i]
        start_train = np.sum(B_arr_train[0:max(t - r + 1, 0)])

        X_train_i = scaler.fit_transform(X_train[start_train:end_train])
        rf.fit(X_train_i, y_train[start_train:end_train])
        xgb.fit(X_train_i, y_train[start_train:end_train])
        
        # compute testing errors
        X_test_i = scaler.transform(X_test[start_test:end_test])
        y_pred_test_RF = rf.predict(X_test_i)
        MSE_test_RF[i] = metrics.mean_squared_error(y_test[start_test:end_test], y_pred_test_RF)
        y_pred_test_XGB = xgb.predict(X_test_i)
        MSE_test_XGB[i] = metrics.mean_squared_error(y_test[start_test:end_test], y_pred_test_XGB)

        # validation
        X_val_i = scaler.transform(X_val[0:end_val])
        y_pred_val_RF = rf.predict(X_val_i)
        losses_RF.append( np.array((y_pred_val_RF - y_val[0:end_val]) ** 2) )
        y_pred_val_XGB = xgb.predict(X_val_i)
        losses_XGB.append( np.array((y_pred_val_XGB - y_val[0:end_val]) ** 2) )
        
        # fixed-window validation
        for j in range(K):
            start_val_j = np.sum(B_arr_val[0:max(t - window_sizes_val[j] + 1, 0)])
            y_pred_val_RF_j = y_pred_val_RF[start_val_j:end_val]
            MSE_val_RF[j, i] = metrics.mean_squared_error(y_val[start_val_j:end_val], y_pred_val_RF_j)
            y_pred_val_XGB_j = y_pred_val_XGB[start_val_j:end_val]
            MSE_val_XGB[j, i] = metrics.mean_squared_error(y_val[start_val_j:end_val], y_pred_val_XGB_j)            
        

    # combine RF and XGB
    MSE_val = np.concatenate((MSE_val_RF, MSE_val_XGB), axis = 1)
    MSE_test = np.concatenate((MSE_test_RF, MSE_test_XGB))
    losses = losses_RF + losses_XGB

    # fixed-window validation
    MSE_fixed = np.zeros(K)
    for j in range(K):
        MSE_fixed[j] = MSE_test[np.argmin(MSE_val[j])]
    
    # ARW
    idx_ARW = tournament_selection(losses, B_arr_val[0:(t+1)], delta, M, seed)
    MSE_ARW = MSE_test[idx_ARW]
    
    return MSE_ARW, MSE_fixed, MSE_test


def test_online(data, list_t, window_sizes_train, window_sizes_val, seed):
    MSE_ARW, MSE_fixed = [], []
    K = len(list_t)
    MSE_ARW, MSE_fixed = np.zeros(K), np.zeros((K, len(window_sizes_val)))
    for (i, t) in enumerate(list_t):
        MSE_ARW[i], MSE_fixed[i], _ = test(data = data, t = t,\
                                    window_sizes_train = window_sizes_train,\
                                    window_sizes_val = window_sizes_val, seed = seed)

    return MSE_ARW, MSE_fixed

