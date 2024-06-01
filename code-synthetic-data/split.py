import numpy as np

def sample_X(**kwargs):
    """subsample from X
    Args:
        X: 1D array in time
        B_arr_train: number of training samples
        B_arr_assess: number of model assessment/selection samples
        seed: random seed
    Returns:
        X_train, X_assess, X_test: 1D arrays
    """
        
    X = kwargs.get('X')
    B_arr = kwargs.get('B_arr')
    B_arr_train = kwargs.get('B_arr_train')
    B_arr_assess = kwargs.get('B_arr_assess')
    seed = kwargs.get('seed')

    y = kwargs.get('y', None)

    num_periods = len(B_arr_train)

    #get the start and end indices for each period
    start_idcs = np.cumsum(B_arr) - B_arr
    end_idcs = np.cumsum(B_arr)

    X_train = []
    X_assess = []
    X_test = []

    if y is not None:
        y_train = []
        y_assess = []
        y_test = []
    np.random.seed(seed)
    #in each period, split data into train, assess and test according to the stipulated B_i's
    for t in range(num_periods):
        start = start_idcs[t]
        end = end_idcs[t]
        X_t = X[start:end]

        #X_train_t, X_assess_t, X_test_t = ThreeWaySplit(X_t, B_arr_train[t], B_arr_assess[t])
        #randomly split indices into train, assess and test
        idx = np.arange(len(X_t))
        #partition the indices into train, assess and test
        idx_train = np.random.choice(idx, B_arr_train[t], replace=False)
        idx_assess = np.random.choice(np.setdiff1d(idx, idx_train), B_arr_assess[t], replace=False)
        idx_test = np.setdiff1d(idx, np.concatenate((idx_train, idx_assess)))
        #get the corresponding data
        X_train.append(X_t[idx_train])
        X_assess.append(X_t[idx_assess])
        X_test.append(X_t[idx_test])

        if y is not None:
            y_t = y[start:end]
            y_train.append(y_t[idx_train])
            y_assess.append(y_t[idx_assess])
            y_test.append(y_t[idx_test])

    if y is not None:
        return np.concatenate(X_train), np.concatenate(X_assess), np.concatenate(X_test), np.concatenate(y_train), np.concatenate(y_assess), np.concatenate(y_test)
    else:
        return np.concatenate(X_train), np.concatenate(X_assess), np.concatenate(X_test)
        
def ThreeWaySplit(X, B1, B2):

    # Randomly shuffle the array
    np.random.shuffle(X)

    # Split into three parts
    part1 = X[:B1]
    part2 = X[B1:B1+B2]
    part3 = X[B1+B2:]

    return part1, part2, part3

