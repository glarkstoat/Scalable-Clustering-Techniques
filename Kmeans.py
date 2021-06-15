import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score as NMI
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime

class Kmeans:
    """
    Calculates the labels resulting from the KMeans algorithm.
    """
    
    def __init__(self, k=153, max_iters=30):
        self.k = k
        self.max_iters = max_iters
        self.losses = []
        self.errors = []
        
    def generateCenters(self, data, random=False):
        """ Generates initial centers as first k points. """        
        
        if random:
            indices = np.random.choice(range(len(data)), self.k, raplace=False)
            return data[indices]
        else:    
            return data[:self.k] 
        
    def fit(self, data):
        self.centers = self.generateCenters(data, random=True)
        self.n = data.shape[0]
        distances = np.zeros((self.n,self.k))

        for _ in tqdm(range(self.max_iters)): 
            
            # Iterates through all clusters
            for i in range(self.k):
                # Calculates squared euclidean distances between every point and every cluster
                distances[:,i] = np.linalg.norm(data - self.centers[i], axis=1)
            
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
            
            # Convergence criterium. If errors are 0 no update was made to any cluster position                     
            error = np.linalg.norm(centers_updated - self.centers)
            if error > 0:
                self.errors.append(error)
                self.centers = centers_updated
            elif error == 0:
                print("\nKmeans converged. Exiting loop.\n")
                return

        self.labels_ = clusters