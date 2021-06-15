import numpy as np
from tqdm import tqdm

class Coreset:
    
    def __init__(self, X):
        self.X = X
        self.mu = np.mean(X, axis=0)

    
    def get_q(self):

        # calculates the q for every sample in the dataset

        self.q = np.zeros(self.X.shape[0])
        #q_sum = np.sum(cdist(self.X, self.mu, metric='sqeuclidean'))
        q_sum = np.sum(np.linalg.norm(self.X - self.mu)**2)
        print(q_sum)  
        X_abs = np.linalg.norm(self.X, ord=np.inf)

        for i, x in tqdm(enumerate(self.X)):
            # calculates the q(x)

            q = 1.0 / (2.0 * X_abs) + 1.0 / 2.0 * \
            (np.linalg.norm(x - self.mu)**2 / q_sum)
            self.q[i] = q

        self.q = self.q.flatten()
        

    def get_coreset(self, m):
        # chooses m random samples 
        rnd_indices = np.random.choice(self.X.shape[0], m, replace=True, p=self.q)

        # calculates the m weighted, randomly selected points
        C = np.zeros((m, self.X.shape[1]))

        for i, idx in enumerate(rnd_indices):
            C[i] = 1 / (m * self.q[idx]) * self.X[idx]

        return C