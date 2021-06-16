from enum import IntEnum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score as NMI
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from scipy.spatial.distance import cdist


class ClusterInitialization(IntEnum):
    random = 0
    firstk = 1
    custom = 2

    def get_centers(self, k=None, data=None, defined=None):
        print(self, self.name, self.value)
        if self.value == ClusterInitialization.random:
            if data is None or k is None:
                raise ValueError
            return data[np.random.choice(range(len(data)), k, replace=False)]
        if self.value == ClusterInitialization.firstk:
            if data is None or k is None:
                raise ValueError
            return data[:k]
        if self.value == ClusterInitialization.custom:
            if defined is None:
                raise ValueError
            return defined


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

    def fit(self, data, cluster_centers=None, init="random"):
        if (cluster_centers is None) and (init == "random"):
            cluster_centers = ClusterInitialization.random.get_centers(k=self.k, data=data)
        if (cluster_centers is None) and (init == "firstk"):
            cluster_centers = ClusterInitialization.firstk.get_centers(k=self.k, data=data)
        self.centers = cluster_centers
        self.n = data.shape[0]

        for _ in tqdm(range(self.max_iters)):

            # Sqaured euclidean distance of every point to every cluster center
            distances = cdist(data, self.centers, 'sqeuclidean')

            # Compute loss i.e. sum of shortest squared distance of each point to all clusters
            self.losses.append(np.sum(np.min(distances, axis=1)) / self.n)

            # Computes indiex of closest cluster for each point. 
            # Functions as computed label for each point.
            clusters = np.argmin(distances, axis=1)
            centers_updated = deepcopy(self.centers)

            # Calculate mean of assigned points for every cluster and update cluster position
            for i in range(self.k):
                if len(data[clusters == i]) == 0:  # no points were assigned to this cluster
                    continue
                else:
                    centers_updated[i] = np.mean(data[clusters == i], axis=0)

            # Measures difference between updated and old cluster centers
            error = np.linalg.norm(centers_updated - self.centers)
            self.errors.append(error)

            # Convergence criterium. If errors are < tol, error small enough to be considered converged                 
            if error > self.tol:
                self.centers = centers_updated
            else:
                print("\nKmeans converged. Exiting loop.\n")
                self.labels_ = clusters
                return

        self.labels_ = clusters
