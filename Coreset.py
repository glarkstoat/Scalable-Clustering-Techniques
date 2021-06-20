import numpy as np
from tqdm import tqdm

class Coreset:
    
    def __init__(self, X):
        """
        Class to calculate the Coreset for a given Dataset
        Parameters:
            X = Dataset (Array of size n x m)
        """

        self.X = X
        self.mu = np.mean(X, axis=0)

    
    def get_q(self):

        # calculates the q for every sample in the dataset
        self.q = np.zeros(self.X.shape[0])
        q_sum = np.sum(np.linalg.norm(self.X - self.mu)**2)

        for i, x in enumerate(self.X):
            # calculates the q(x)
            q = 0.5 * (1/self.X.shape[0]) + 0.5 * (np.linalg.norm(x - self.mu)**2 / q_sum)

            self.q[i] = q
        

    def get_coreset(self, m):
        """
        Returns the coreset, the randomly chosen indices for the samples
        and generates the weights to use in sklearn.
        Parameters:
            m = number of samples in the coreset
        """

        # chooses m random samples 
        rnd_indices = np.random.choice(self.X.shape[0], m, p=self.q)

        # calculates the m weighted, randomly selected points
        C = np.zeros((m, self.X.shape[1]))
        self.weights = []

        for i, idx in enumerate(rnd_indices):
            self.weights.append(1 / (m * self.q[idx]))
            C[i] = 1 / (m * self.q[idx]) * self.X[idx]

        return C, rnd_indices

    
    def get_m(self, k=153, eps=0.5, delta=0.5):

        """
        Returns the m using Theorem 2 of the paper.
        Parameters:
            k = Number of Clusters in the Data
            eps = Float between (0, 1)
            delta = Float between (0, 1)
        """

        return int((self.X.shape[1] * k * np.log(k) + np.log(1/delta)) / eps**2)