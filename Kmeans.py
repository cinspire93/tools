import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

class Kmeans(object):

    '''
    Expects rows in data to be data points, and columns to be different features
    '''

    def __init__(self, max_iterations=10000, num_centroids=3, dist_metric='euclidean',
                 smart_centroid_init=True):
        '''
        Takes in a max number of iterations,
        number of clusters desired,
        and a distance metric ['euclidean', 'cosine', 'manhattan']
        '''
        self._dist_funcs = {'euclidean': euclidean_distances,
                            'cosine': cosine_similarity,
                            'manhattan': manhattan_distances}
        self.max_iterations = max_iterations
        self.num_centroids = num_centroids
        self.smart = smart_centroid_init
        self.centroids = []
        self.dist_mat = None
        self.assignments = None
        self._dist_func = self._dist_funcs[dist_metric]

    def _distance(self, data):
        '''
        Takes in 2d np array data
        Sets a 2d np array of distances, with (i, j)th element representing
        the distance between ith centroid and jth data point
        '''
        self.dist_mat = self._dist_func(self.centroids, data)

    def _sample_centroids(self, data):
        '''
        Takes in 2d np array data
        Sets a 2d np array of centroids randomly sampled from the data
        '''
        if self.smart:
            self._smart_centroid_init(data)
        else:
            self.centroids = np.array(random.sample(data, self.num_centroids))

    def _smart_centroid_init(self, data):
        '''
        If the smart centroid init option is True,
        Takes in 2d np array data
        Sets centroids according to the Kmeans++ algorithm
        '''
        centroids_loc = []
        sample_size = data.shape[0]
        weights = np.array([1.0 / sample_size] * sample_size)
        for i in xrange(self.num_centroids):
            centroids_loc.append(data[np.random.choice(sample_size, p=weights)])
            self.centroids = np.array(centroids_loc)
            dist2 = np.min(self._dist_func(self.centroids, data)**2, axis=0)
            weights = dist2 / np.sum(dist2)

    def _update_centroid(self, data):
        '''
        Takes in 2d np array data
        Update in-class centroids and Provide data point assignments
        '''
        self._distance(data)
        self.assignments = np.argmin(self.dist_mat, axis=0)
        for centroid_index in xrange(self.num_centroids):
            self.centroids[centroid_index] = np.mean(data[self.assignments==centroid_index], axis=0)

    def fit(self, data):
        '''
        Takes in 2d np array data
        Run KMeans algorithm
        '''
        self._sample_centroids(data)
        for iterations in xrange(self.max_iterations):
            prev_centroids = self.centroids
            self._update_centroid(data)
            if np.sum(np.linalg.norm(self.centroids-prev_centroids, axis=1)) < 0.00001:
                break
