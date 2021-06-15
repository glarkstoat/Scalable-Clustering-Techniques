import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

class Coreset:
    
    def __init__(self, X):
        self.X = X
        self.mu = np.mean(X, axis=0, keepdims=True)

    def dist(self, x, y):
        # Took this formulation from the sklearn euclidean distance documentation
        # Deprecated, replaced by scipy cdist with sqeuclidean
        return np.sqrt(np.dot(x, y) - 2 * np.dot(x,y) + np.dot(y, y))**2
    
    def get_q(self):

        # calculates the q for every sample in the dataset

        self.q = []
        q_sum = np.sum(cdist(self.X, self.mu, metric='sqeuclidean'), keepdims=True)  
        print(q_sum)

        X_abs = np.abs(self.X)


        for x in tqdm(self.X):
            # calculates the q(x)
            self.q.append( 1.0 / (2.0 * X_abs) + 1.0 / 2.0 * \
                            (cdist(x.reshape((1,74)), self.mu, metric='sqeuclidean') / q_sum) )
        
        return self.q

    def get_coreset(self, m):

        # chooses m random samples 
        idx = np.random.choice(self.X.shape[0], m, replace=True, p=self.q)

        # calculates the m weighted, randomly selected points
        C = 1 / (m * self.q[idx]) * self.X[idx]

        return C


data = np.genfromtxt("data/bio_train.csv", delimiter=",")[:,3:]
labels_true = np.genfromtxt("data/bio_train.csv", delimiter=",")[:,0]

coreset = Coreset(data)
q = coreset.get_q()