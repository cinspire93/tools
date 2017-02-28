import numpy as np

'''
All of the following distance functions take in 2d array data and 1d array v
Returns a 1d array of distances between each data point and v
We assume X's rows to be data points and columns to be features
'''
def euc_dist(X, v):
    return np.sqrt(np.sum((X-v)**2, axis=0))

def cos_dist(X, v):
    return 1 - X.dot(v) / (np.linalg.norm(X, axis=1) * np.linalg.norm(v))

def man_dist(X, v):
    return np.sum(np.abs(X-v), axis=1)

class Knn(object):

    '''
    Expects the data to have rows as observations and columns as features
    '''

    def __init__(self, k=3, dist_metric='euclidean'):
        '''
        Takes in desired number of neighbors as k
        desired distance metric ['euclidean', 'cosine', 'manhattan']
        '''
        self._dist_funcs = {'euclidean': euc_dist,
                            'cosine': cos_dist,
                            'manhattan': man_dist}
        self.k = k
        self.predictions = []
        self._dist_func = self._dist_funcs[dist_metric]

    def fit(self, data, labels):
        '''
        Fits the model using train data and labels
        '''
        self.X = data
        self.y = labels

    def _predict_one(self, datapoint):
        '''
        Takes in 1d np array datapoint
        Run the Knn algorithm and predict label for the new observation
        '''
        neighbors_idx = np.argsort(self._dist_func(self.X, datapoint))[:self.k]
        neighbors_labels = self.y[neighbors_idx]
        ulabels, counts = np.unique(neighbors_labels, return_counts=True)
        return ulabels[np.argmax(counts)]

    def predict(self, new_data):
        '''
        Takes in 2d np array new_data
        Predict labels for all new observations
        '''
        for datapoint in new_data:
            self.predictions.append(self._predict_one(datapoint))
        return np.array(self.predictions)

    def score(self, y_predict, y_test):
        '''
        Takes in 1d array of predicted labels and 1d array of actual labels
        Returns accuracy and sensitivity of Knn
        '''
        accuracy = np.sum(y_predict==y_test)*1.0 / len(y_test)
        sensitivity = np.sum(y_predict*y_test==1) / len(y_test)
        return accuracy, sensitivity
