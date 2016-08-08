from __future__ import division
import numpy as np

class NMF:
    def __init__(self, rating, dimension, steps=30000, alpha=0.0002, beta=0.02, threshold=0.001):
        self.rating = rating
        self.p = np.random.rand(rating.shape[0], dimension)
        self.q = np.random.rand(dimension, rating.shape[1])
        self.alpha = alpha
        self.beta = beta
        self.steps = steps
        self.threshold = threshold

    def get_error(self):
        rating_error = np.sum((self.rating - np.dot(self.p, self.q)) ** 2)
        regularize = self.beta / 2.0 * (np.linalg.norm(self.p) ** 2 + np.linalg.norm(self.q) ** 2)
        return rating_error + regularize

    def update(self):
        error = self.rating - np.dot(self.p, self.q)
        error[np.where(self.rating == 0)] = 0
        temp_q = self.q.copy()
        self.q += 2 * self.alpha * np.dot(self.p.T, error)
        self.p += 2 * self.alpha * np.dot(error, temp_q.T)

    def run(self):
        for i in xrange(self.steps):
            self.update()
            error = self.get_error()
            if(i % 1000 == 0):
                print error
            if(error < self.threshold):
                break
        return np.dot(self.p, self.q)

if __name__ == '__main__':
    R = np.array([[5, 3, 0, 1],[4, 0, 0, 1],[1, 1, 0, 5],[1, 0, 0, 4],[0, 1, 5, 4]])
    mf = NMF(rating=R, dimension=2)
    nR = mf.run()
    print R
    print nR
