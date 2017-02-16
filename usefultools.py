import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as import pd
import scipy.stats as scs
from sklearn.metrics import confusion_matrix

def residual_plot(model, X, y):
    '''
    Takes in a specified sklearn model, X and y
    Returns a residual plot
    '''
    model.fit(X, y)
    resids = model.resid
    resid_fig = plt.figure(figsize=(8, 5))
    plt.scatter(y, resids, edgecolors='none', color='r', alpha=0.6, label='residuals')
    x_axis = sorted(y)
    plt.plot((), (0, 0), linewidth=2, linestyle='--', color='g')
    plt.title('Residual Plot', fontsize=15)
    plt.xlabel('Actual Observation', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.legend()

def samp_size_ci(width, alpha, sigma):
    '''
    Takes in the full desired width of the CI,
    deisred alpha, current standard deviation
    Returns the minimum sample size required
    '''
    z_alpha = scs.norm.ppf(1 - alpha/2.0)
    return (2 * z_alpha * sigma / width)**2

def samp_size_power(alpha, beta, miu_0, miu_a, sigma, two_tailed=True):
    '''
    Takes in desired alpha, beta,
    hypothesized miu, alternative miu, standard deviation
    Returns the minimum sample size required
    '''
    if two_tailed:
        alpha = alpha / 2.0
    z_beta, z_alpha = scs.norm.ppf(1 - beta), scs.norm.ppf(1 - alpha)
    min_n = ((z_alpha+z_beta) * sigma / (miu_a-miu_0))**2
    return math.ceil(min_n)
