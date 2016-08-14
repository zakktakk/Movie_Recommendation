from __future__ import division
import numpy as np

class MatrixFactorization:
    #https://datajobs.com/data-science-repo/Collaborative-Filtering-[Koren-and-Bell].pdf
    def __init__(self, rating, dimension, steps=5000, gamma=0.005, lmd=0.02, threshold=0.001):
        self.rating = rating
        self.mu = np.mean(rating[rating > 0])
        self.b_u = self.b_u()
        self.b_i = self.b_i()
        self.p = np.random.rand(rating.shape[0], dimension)
        self.q = np.random.rand(dimension, rating.shape[1])
        self.gamma = gamma
        self.lmd = lmd
        self.steps = steps
        self.threshold = threshold

    def b_u(self):
        ret = np.mean(self.rating, axis=1, keepdims=True)
        items = self.rating.shape[1]
        for row in xrange(ret.shape[0]):
            nonzero = np.count_nonzero(self.rating[row])
            ret[row] *= items / nonzero
        return ret - self.mu

    def b_i(self):
        ret = np.mean(self.rating, axis=0)
        users = self.rating.shape[0]
        for col in xrange(ret.shape[0]):
            nonzero = np.count_nonzero(self.rating[:, col])
            ret[col] *= users / nonzero
        return ret - self.mu

    def get_bias(self):
        return np.zeros((self.rating.shape[0], self.rating.shape[1])) + self.mu + self.b_u + self.b_i

    def get_error(self):
        rating_error = np.sum((self.rating - np.dot(self.p, self.q) - self.get_bias()) ** 2)
        regularize = np.linalg.norm(self.p) ** 2 + np.linalg.norm(self.q) ** 2 + np.linalg.norm(self.b_u) ** 2 + np.linalg.norm(self.b_i) ** 2
        return rating_error + self.lmd * regularize

    def update(self):
        error = self.rating - np.dot(self.p, self.q) - self.get_bias()
        error[np.where(self.rating == 0)] = 0
        temp_q = self.q.copy()
        self.q += self.gamma * (np.dot(self.p.T, error) - self.lmd * self.q)
        self.p += self.gamma * (np.dot(error, temp_q.T) - self.lmd * self.p)
        self.b_u += self.gamma * (np.mean(error, axis=1, keepdims=True) - self.lmd * self.b_u)
        self.b_i += self.gamma * (np.mean(error, axis=0) - self.lmd * self.b_i)

    def run(self):
        for i in xrange(self.steps):
            self.update()
            error = self.get_error()
            if(i % 1000 == 0):
                print error
            if(error < self.threshold):
                break
        return np.dot(self.p, self.q) + self.get_bias()

if __name__ == '__main__':
    R = np.array([[5, 3, 0, 1],[4, 0, 0, 1],[1, 1, 0, 5],[1, 0, 0, 4],[0, 1, 5, 4]])
    mf = MatrixFactorization(rating=R, dimension=2)
    nR = mf.run()
    print R
    print nR
