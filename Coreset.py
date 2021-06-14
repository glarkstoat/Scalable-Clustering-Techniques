import numpy as np

class Coreset:
    
    def __init__(self, X, m):
        self.X = X
        self.m = m
        self.mu = np.mean(X, axis=1)

    def dist(x, y):
        return np.sqrt(np.dot(x, y) - 2 * np.dot(x,y) + np.dot(y, y))**2
    
    def fit_coreset(self):

        self.q = []

        for x in self.X:
            self.q.append( 1.0 / (2.0 * np.abs(self.X)) + 1.0 / 2.0 * \
                            (self.dist(x, self.mu) / np.sum(self.dist(self.X, self.mu), axis=1)) )
        
        return self.q