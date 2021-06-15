import numpy as np
from tqdm import tqdm

class Coreset:
    
    def __init__(self, X):
        self.X = X
        self.mu = np.mean(X, axis=0)

    def dist(self, x, y):
        # Took this formulation from the sklearn euclidean distance documentation
        return np.sqrt(np.dot(x, y) - 2 * np.dot(x,y) + np.dot(y, y))**2
    
    def get_q(self):

        # calculates the q for every sample in the dataset

        self.q = []
        q_sum = np.sum(self.dist(self.X, self.mu))  # produces a sum of NaN, needs fixing
                                                    # possibly in dist function
        X_abs = np.abs(self.X)


        for x in tqdm(self.X):
            # calculates the q(x)
            self.q.append( 1.0 / (2.0 * X_abs) + 1.0 / 2.0 * \
                            (self.dist(x, self.mu) / q_sum) )
        
        return self.q

    def get_coreset(self, m):

        # chooses m random samples 
        idx = np.random.choice(self.X.shape[0], m, replace=False)

        # calculates the m weighted, randomly selected points
        C = 1 / (m * self.q[idx]) * self.X[idx]

        return C


data = np.genfromtxt("data/bio_train.csv", delimiter=",")[:,3:]
print(data.shape)
labels_true = np.genfromtxt("data/bio_train.csv", delimiter=",")[:,0]

coreset = Coreset(data)
q = coreset.get_q()