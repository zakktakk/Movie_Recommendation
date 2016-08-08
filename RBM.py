from __future__ import division
import numpy as np

class RBM:
    def __init__(self, n_visible, n_hidden):
        self.W = np.random.uniform(low=-0.08, high=0.08, size=(n_visible, n_hidden)).astype("float32")
        self.b_v = np.zeros(n_visible).astype("float32")
        self.b_h = np.zeros(n_hidden).astype("float32")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #p(v=1|h,o)
    def prob_v(self, h):
        return self.sigmoid(self.b_v + np.dot(self.W, h))

    def prob_h(self, v):
        return self.sigmoid(self.b_h + np.dot(v.T, self.W))

    
