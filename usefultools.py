import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as import pd
import scipy.stats as scs
from sklearn.metrics import confusion_matrix


def standard_confusion_matrix(labels, y_predict):
    '''
    Returns the confusion matrix as a 2x2
    numpy array of the form [[tp fn],[fp tn]].
    '''
    [[tn, fp],[fn, tp]] = confusion_matrix(y_true, y_predict)
    cm = np.array([[tp, fn],[fp, tn]])
    cm = cm / float(cm.sum())

    return cm

def profit_curve(cb, labels, predict_probas):
    '''
    Takes in numpy 2d arra of costbenefit matrix,
    np array of true class labels,
    np array of predicted probabilities
    Returns a list of expected profit at each threshold
    '''
    # Reverse the list of predicted_probs
    thresholds = sorted(predict_probas, reverse=True)

    expected_profits = []
    for threshold in thresholds:
        y_predict = predict_probas > threshold
        cm = standard_confusion_matrix(labels, y_predict)
        expected_profits.append((cm*cb).sum())

    return expected_profits

def plot_profit_curve(model, plotlabel, cb, X_train, X_test, y_train, y_test):
    '''
    Takes in a sklearn model, str of plotlabel,
    numpy 2d array of costbenefit matrix,
    X_train, X_test, y_train, y_test
    Returns a graph of the profit curve associated with that model
    '''
    model.fit(X_train, y_train)
    # Predicted probability has two columns, we want the second one
    y_predict_probas = model.predict_proba(X_test)[:,1]

    expected_profits = profit_curve(cb, y_test, y_predict_probas)
    percentages = np.arange(0, 100, 100. / len(profits))

    plt.plot(percentages, expected_profits, label=plotlabel)
    plt.title("Profit Curve")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.legend(loc='lower left')

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
    z_alpha = scs.norm.ppf(1-alpha/2.0)
    return (2*z_alpha*sigma/width)**2

def samp_size_power(alpha, beta, miu_0, miu_a, sigma, two_tailed=True):
    '''
    Takes in desired alpha, beta,
    hypothesized miu, alternative miu, standard deviation
    Returns the minimum sample size required
    '''
    if two_tailed:
        alpha = alpha/2.0
    z_beta, z_alpha = scs.norm.ppf(1-beta), scs.norm.ppf(1-alpha)
    min_n = ((z_alpha+z_beta)*sigma/(miu_a-miu_0))**2
    return math.ceil(min_n)
