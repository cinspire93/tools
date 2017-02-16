import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

def calc_expected_profit(costbenefit_matrix, predict_probs, labels):
    '''
    Input: costbenefit_matrix, predicted probabilities, true labels
    Output: expected profit given the costbenefit_matrix
    P.S. cost benefit matrix must be in sklearn confusion matrix format
    '''
    # Reverse the list of predicted_probs
    thresholds = sorted(predict_probs, reverse=True)
    expected_profits = []
    for threshold in thresholds:
        y_predict = predict_probs > threshold
        confuse_matrix = confusion_matrix(labels, y_predict)
        expected_profits.append((confuse_matrix * costbenefit_matrix).sum())
    return expected_profits

class ClassifierRunner(object):

    def __init__(self, classifiers_lst):
        '''
        Takes a list of initialized Classifiers from sklearn
        Initializes a internal list of classifiers and a list of their names
        '''
        self.classifiers = classifiers_lst
        self.classifier_names = [clf.__class__.__name__ for clf in self.classifiers]

    def training(self, X, y):
        '''
        Takes in X and y,
        Train every model in the list on X and y
        '''
        for clf in self.classifiers:
            clf.fit(X, y)

    def cross_val_scoring(self, X_train, y_train, folds=5, metric='accuracy'):
        '''
        Takes in number folds in cross validation, desired metric, X and y
        Gives a dictionary with modle names as keys, and their scores as values
        '''
        scores = np.array([cross_val_score(clf, X_train, y_train, cv=folds, scoring=metric, n_jobs=-1).mean()
                           for clf in self.classifiers])
        self.cross_val_scores = dict(zip(self.classifier_names, scores))

    def plot_roc_curve(self, X_test, y_test):
        roc_figure = plt.figure(figsize=(10, 10))
        for clf_name, clf in zip(self.classifier_names, self.classifiers):
            predicted_probs = clf.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, predicted_probs, pos_label=1)
            roc_auc = auc(x=fpr, y=tpr)
            plt.plot(fpr, tpr, label='{}-AUC: {:.2f}'.format(clf_name, roc_auc), lw=5)
        diag = np.linspace(0, 1.0, 25)
        plt.plot(diag, diag, color='grey', lw=3, ls='--')
        plt.legend(loc='best', fontsize=20)
        plt.title('ROC curves for all models', fontsize=35)
        plt.xlabel('False Positive Rate', fontsize=25)
        plt.ylabel('True Positive Rate', fontsize=25)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.savefig('ROC_curves.png', dpi=100)
        plt.show()

    def plot_profit_curve(self, X_test, y_test, costbenefit_matrix):
        profit_figure = plt.figure(figsize=(10, 10))
        for clf_name, clf in zip(self.classifier_names, self.classifiers):
            predicted_probs = clf.predict_proba(X_test)[:, 1]
            expected_profits = calc_expected_profit(costbenefit_matrix, predicted_probs, y_test)
            percentages = np.arange(0, 100, 100./len(expected_profits))
            plt.plot(percentages, expected_profits, label=clf_name, lw=5)
        plt.legend(loc='best', fontsize=20)
        plt.title('Profit curves for all models', fontsize=35)
        plt.xlabel('Percentage of Test Instances(decreasing by score)', fontsize=25)
        plt.ylabel('Profit', fontsize=25)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.savefig('Profit_curves.png', dpi=100)
        plt.show()
