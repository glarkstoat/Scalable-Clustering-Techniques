import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score as NMI
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from scipy.spatial.distance import cdist

class Kmeans:
    """
    Calculates the labels resulting from the KMeans algorithm.
    """
    
    def __init__(self, k=153, max_iters=30, tol=0.0001):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.losses = []
        self.errors = []
        
    def generateCenters(self, data):
        """ Generates initial centers as first k points. """      
        indices = np.random.choice(range(len(data)), self.k, replace=False)
        return data[indices]
        
    def fit(self, data, cluster_centers=None):
        self.centers = self.generateCenters(data) if cluster_centers is None else cluster_centers
        self.n = data.shape[0]

        for _ in tqdm(range(self.max_iters)): 
            
            # Sqaured euclidean distance of every point to every cluster center
            distances = cdist(data, self.centers, 'sqeuclidean')
            
            # Compute loss i.e. sum of shortest squared distance of each point to all clusters
            self.losses.append(np.sum(np.min(distances, axis=1))/self.n)
            
            # Computes indiex of closest cluster for each point. 
            # Functions as computed label for each point.
            clusters = np.argmin(distances, axis = 1)
            centers_updated = deepcopy(self.centers)
            
            # Calculate mean of assigned points for every cluster and update cluster position
            for i in range(self.k):
                if len(data[clusters == i]) == 0: # no points were assigned to this cluster
                    pass
                else:
                    centers_updated[i] = np.mean(data[clusters == i], axis=0)
         
            # Convergence criterium. If errors are < tol, error small enough to be considered converged                 
            error = np.linalg.norm(centers_updated - self.centers)
            if error > self.tol:
                self.errors.append(error)
                self.centers = centers_updated
            else:
                print("\nKmeans converged. Exiting loop.\n")
                self.labels_ = clusters
                return
        
        self.labels_ = clusters

