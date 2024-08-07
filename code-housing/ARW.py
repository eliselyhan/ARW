import numpy as np


# Compute summary statistics

def prepare(U, B_arr):

    """Compute B_tk, mu_hat_tk and v_hat_tk for all k
    Args:
        U (np.array): 1D array of length B_1 + ... + B_t
        B_arr: (np.array) stores the sample size for each period
    """

    period_ends = np.cumsum(B_arr)
    period_starts = period_ends - B_arr
    t = len(B_arr)

    # Compute first and second moments for each period, stored in reverse order
    moments_1 = np.array( [np.mean( U[period_starts[i]:period_ends[i]] ) for i in range(t)] )[::-1]
    moments_2 = np.array( [np.mean( U[period_starts[i]:period_ends[i]] ** 2 ) for i in range(t)] )[::-1]

    # Compute mu_hat_tk and its second-moment version
    B_rev = B_arr[::-1] # reverse
    B_all = np.cumsum(B_rev) # the k-th entry is B_{t, k+1}
    mu_hat_all = np.cumsum( moments_1 * B_rev ) / B_all # the k-th entry is mu_hat_{t, k+1}
    mu_2_hat_all = np.cumsum( moments_2 * B_rev ) / B_all

    # Compute v_hat_tk
    if B_all[0] > 1:
        tmp = B_all / (B_all - 1)
    else: # B_all[0] == 1
        tmp = np.zeros(t)
        if t > 1:
            tmp[1:] = B_all[1:] / (B_all[1:] - 1)
    v_hat_all = np.sqrt( tmp * (mu_2_hat_all - mu_hat_all ** 2) ) # the k-th entry is v_hat_{t, k+1}

    return B_all, mu_hat_all, v_hat_all



# Model evaluation

def ARWME(U, B_arr, delta = 0.1, M = 0):
    """ Selecting the best window size
    Args:
        U (np.array): 1D array of length B_1 + ... + B_t
        B_arr: (np.array) stores the sample size for each period
        delta, M (float): tuning parameters

    Returns:
        best_window (int): best window size
        best_mean (float): estimated population mean
    """

    # Compute B_tk, mu_tk and v_tk for all k
    B_all, mu_hat_all, v_hat_all = prepare(U, B_arr)
    t = len(B_all)

    # Compute psi_hat_k for all k
    psi_hat = np.empty(t)
    for k in range(t):
        B_tk = B_all[k]
        if B_all[k] == 1:
            psi_hat[k] = M
        else:
            psi_hat[k] = v_hat_all[k] * np.sqrt( 2 * np.log(2/delta) / B_tk ) + 8 * M * np.log(2/delta) / ( 3 * (B_tk-1) )
    
    # Compute phi_hat_k for all k
    phi_hat = np.empty(t)
    for k in range(t):
        tmp = np.array([ np.abs(mu_hat_all[k] - mu_hat_all[i]) - psi_hat[i] - psi_hat[k] for i in range(k + 1) ])
        tmp[tmp < 0] = 0
        phi_hat[k] = np.max(tmp)
 
    # Final steps
    k_hat = np.argmin(phi_hat + psi_hat) + 1 
    return k_hat, mu_hat_all[k_hat - 1]



# Pairwise comparison

def pairwise_comparison(loss_1, loss_2, B_arr, delta = 0.1, M = 0):
    
    """ Pairwise comparison between two models
    Args:
        loss_1, loss_2 (np.array): 1D arrays of length B_1 + ... + B_t
        B_arr: (array) stores the sample size for each period
        delta, M (float): tuning parameters

    Returns:
        result (int): 1 if model_1 wins, 2 if model_2 wins
    """
    
    # Estimate the performance gap
    _, delta_hat = ARWME(U = loss_1 - loss_2, B_arr = B_arr, delta = delta, M = M)

    # If estimated gap < 0, return 1; otherwise, return 2
    if delta_hat < 0:
        return 1
    else:
        return 2



# Model selection through a single-elimination tournament

def tournament_selection(losses, B_arr, delta = 0.1, M = 0, seed = 2024):

    """ select the best model through a tournament 

    Args:
        losses (list of np.arrays)): len = m; each element is a 1D array of length B_1 + ... + B_t
        B_arr (array): stores the sample size for each period
        delta, M (float): tuning parameters
        seed (int): random seed

    Returns:
        best_model_index (int): index of the best model
    """
    
    np.random.seed(seed)
    num_models = len(losses)
    
    # Pair up the models; compare each pair and choose the winner; advance to the next round
    candidates = list(np.arange(num_models))

    while num_models > 1:

        #shuffle
        np.random.shuffle(candidates)
        
        #create a list of winners that advance to the next round
        new_candidates = []

        #select bye-team if odd number of models
        if num_models % 2 == 1:
            new_candidates.append( candidates[-1] )
            del candidates[-1]
                    
        #pair up the remaining models
        for i in range(int(num_models/2)):
            model_1, model_2 = candidates[2*i], candidates[2*i+1]
            result = pairwise_comparison(loss_1 = losses[model_1], loss_2 = losses[model_2], B_arr = B_arr, delta = delta, M = M)
            if result == 1:
                new_candidates.append(model_1)
            else:
                new_candidates.append(model_2)

        candidates = new_candidates
        num_models = len(candidates)
        
    return int(candidates[0])





