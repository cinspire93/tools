import pandas as import pd
import numpy as np
import matplotlib.pyplot as plt
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
