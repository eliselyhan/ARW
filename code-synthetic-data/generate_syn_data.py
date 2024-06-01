import numpy as np
import matplotlib.pyplot as plt

def generate_gaussian_sequence(B_arr, true_means, variance):

    # Generate random data points for each period
    X_list = [np.random.normal(loc=mu, scale=variance, size=B_i) for mu, B_i in zip(true_means, B_arr)]

    # Combine assesment data into a single 1D array
    X_1D = np.concatenate(X_list)

    return X_1D


def generate_syn_data(N, n, seed=2024):
    
    theta = np.zeros(1 + N)
    np.random.seed(seed)

    # phase 1: big shifts
    eta = 0.005
    le = 1
    tmp = le
    for i in range(2 * n):
        theta[le + i] = theta[le + i - 1] + eta
        tmp += 1
    le = tmp
    for i in range(int(n / 2)):
        theta[le + i] = theta[le + i - 1] - eta
        tmp += 1
    le = tmp
    for i in range(int(n / 2)):
        theta[le + i] = theta[le + i - 1]
        tmp += 1
    le = tmp

    # phase 2: sinusoid
    for i in range(2 * n):
        theta[le + i] = theta[le - 1] - 0.1 * np.sin( np.pi * i / (1 * n) )
        tmp += 1
    le = tmp
    for i in range(2 * n):
        theta[le + i] = theta[le - 1] - 0.1 * np.sin( np.pi * i / (3 * n) )
        tmp += 1
    le = tmp
    print(le)

    # phase 3: constant
    for i in range(8 * n):
        theta[le + i] = theta[le - 1] - 0.3
        tmp += 1
    le = tmp

    # phase 4: steps
    m = N - le + 1
    print(m)
    steps = np.random.binomial(1, 0.5, size = m) * 2 - 1
    for i in range(m):
        theta[le + i] = theta[le + i - 1] + 0.02 * steps[i]
        tmp += 1
    le = tmp

    #plt.plot(theta, 'k-')
    return theta