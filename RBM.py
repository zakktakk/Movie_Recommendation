from __future__ import division
import numpy as np

class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=0.1, epochs=1000):
        self.W = np.random.uniform(low=-0.08, high=0.08, size=(n_visible, n_hidden)).astype("float32")
        self.b_v = np.zeros(n_visible).astype("float32")
        self.b_h = np.zeros(n_hidden).astype("float32")
        self.learning_rate = learning_rate
        self.epochs = epochs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_x = np.exp(x)
        exp_sum = np.sum(exp_x, axis=1, keepdims=True)
        return exp_x / exp_sum

    #p(v=1|h)
    def prob_v(self, h):
        return self.softmax(self.b_v + np.dot(self.W, h))

    #p(h=1|v)
    def prob_h(self, v):
        return self.sigmoid(self.b_h + np.dot(v.T, self.W))
