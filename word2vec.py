#this is class for ward2vec using CBoW method (not implementing skip-gram)
from __future__ import division
import numpy as np
import math
import sys
#Dense Layer
class Layer:
    # Constructor
    def __init__(self, in_dim, out_dim, function, deriv_function):
        self.W = np.random.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim)).astype("float32")
        self.b = np.zeros(out_dim).astype("float32")
        self.function = function
        self.deriv_function = deriv_function
        self.u = None
        self.delta = None
        self.r_dW = np.zeros(out_dim)
        self.r_db = np.zeros(out_dim)

    #Forward Propagation
    def f_prop(self, x):
        self.u = np.dot(x, self.W) + self.b
        self.z = self.function(self.u)
        return self.z

    #Back Propagation
    def b_prop(self, delta, W):
        self.delta = np.dot(delta, W.T) * self.deriv_function(self.u)
        return self.delta

#Projection Layer
class Projection:
    def __init__(self, in_dim, out_dim, scale):
        rng = np.random.RandomState(1234)
        self.W = rng.randn(in_dim, out_dim) * scale
        self.delta = None
        self.r_dW = np.zeros(out_dim)

    def f_prop(self, x):
        self.z = np.sum(self.W[x], axis = 1) / len(x[0])
        return self.z

    def b_prop(self, delta, W):
        self.delta = np.dot(delta, W.T)
        return self.delta

#define activation function
def softmax(x):
    exp_x = np.exp(x)
    exp_sum = np.sum(exp_x, axis = 1, keepdims = True)
    return exp_x / exp_sum

#define deriv activation function
def deriv_softmax(x):
    return softmax(x) * (1 - softmax(x))

#CBoW
def CBoW(sentenses, window_size=2):
    #make train data from sentenses
    def make_train_from_sentenses(sentenses):
        ret = []
        for i, sentense in enumerate(sentenses):
            for j in xrange(window_size, len(sentense)-window_size):
                data_range = range(j - window_size, j + window_size + 1)
                data_range.remove(j)
                train_data = sentense[data_range]
                ret.append(train_data)
        return ret

    #make correct one hot vector from sentence
    def make_onehot(sentenses):
        ret = []
        for sentense in sentenses:
            temp = sentense[window_size:len(sentense)-window_size]
            for i in xrange(len(temp)):
                one_hot_code = np.zeros(item_num)
                one_hot_code[temp[i]] = 1
                ret.append(one_hot_code)
        return ret

    #Forward Propagation of multilayer
    def f_props(layers, x):
        z = x
        for layer in layers:
            z = layer.f_prop(z)
        return z
    #Back Propagation of multilayer
    def b_props(layers, delta):
        for i, layer in enumerate(layers[::-1]):
            if i == 0:
                layer.delta = delta
            else:
                delta = layer.b_prop(delta, _W)
            _W = layer.W
    #define error function
    def er_cost(y, t):
        return np.sum(np.log(y) * t)

    def train(X, t, alpha=1.0e-2, eps=1.0e-8):
        #Forward Propagation
        y = f_props(layers, X)
        #Cost Function & delta
        cost = er_cost(y, t)
        delta = y - t
        #Back Propagation
        b_props(layers, delta)
        #Update Parameters
        z = X
        for i, layer in enumerate(layers):
            if i == 0:
                dW = layer.delta
                layer.r_dW = layer.r_dW + dW ** 2
                layer.W[z] = layer.W[z] - alpha / (np.sqrt(layer.r_dW) + eps) * dW
            else:
                dW = np.dot(z.T, layer.delta)
                db = np.dot(np.ones(len(z)), layer.delta)
                layer.r_dW = layer.r_dW + dW ** 2
                layer.r_db = layer.r_db + db ** 2
                layer.W = layer.W - alpha / (np.sqrt(layer.r_dW) + eps) * dW
                layer.b = layer.b - alpha / (np.sqrt(layer.r_db) + eps) * db
            z = layer.z
        return cost

    #movie user rating info
    item_num = 1682
    layers = [Projection(item_num, 100, 1.0), Layer(100, item_num, softmax, deriv_softmax)]
    #make dataset from sentenses
    train_X = make_train_from_sentenses(sentenses)
    train_y = make_onehot(sentenses)
    for epoch in xrange(3):
        for x,y in zip(train_X, train_y):
            cost = train(x[np.newaxis, :], y[np.newaxis, :])
    return layers[0].W
