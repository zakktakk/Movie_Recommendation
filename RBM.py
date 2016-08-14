from __future__ import division
import numpy as np

class RBM:
    def __init__(self, n_visible, n_hidden, rating, rating_num=5, learning_rate=0.1, t=5, lmd=0.0001, epochs=1000):
        self.rating_num = rating_num
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.init_weight()
        self.init_visible_bias()
        self.init_hidden_bias()
        self.learning_rate = learning_rate
        self.t = t
        self.users = input.shape[0]
        self.input = self.convert_input(rating)
        self.lmd = lmd
        self.epochs = epochs

    def init_weight(self):
        self.W_dict = {}
        for step in xrange(1,self.rating_num+1):
            W = np.random.uniform(low=-1/self.n_visible, high=1/self.n_visible, size=(self.n_visible, self.n_hidden)).astype("float32")
            self.W_dict.update({step:W})

    def init_visible_bias(self):
        self.b_v_dict = {}
        for step in xrange(1,self.rating_num+1):
            b_v = np.zeros(self.n_visible).astype("float32")
            self.b_v_dict.update({step:b_v})

    def init_hidden_bias(self):
        self.b_h_dict = {}
        for step in xrange(1,self.rating_num+1):
            b_h = np.zeros(self.n_hidden).astype("float32")
            self.b_h_dict.update({step:b_h})

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #p(v=1|h)
    def prob_v_reconstruct(self, h_dict):
        prob_v_dict = {}
        prob_v_sum = np.zeros((self.users, self.n_visible)).astype("float32")
        for step in xrange(1,self.rating_num+1):
            val = np.exp(self.b_v_dict[step] + np.dot(h_dict[step], self.W_dict[step].T))
            prob_v_dict.update({step:val})
            prob_v_sum += val
        prob_v_dict.update((key, prob_v_dict[key] / prob_v_sum) for key in prob_v_dict)
        return prob_v_dict

    def prob_v_dict(self, h_dict):
        prob_v_dict = {}
        for step in xrange(1, self.rating_num+1):
            val = self.sigmoid(self.b_v_dict[step] + np.dot(h_dict[step], self.W_dict[step].T))
            prob_v_dict.update({step:val})
        return prob_v_dict

    #p(h=1|v)
    def prob_h_dict(self, v_dict):
        prob_h_dict = {}
        for step in xrange(1,self.rating_num+1):
            val = self.sigmoid(self.b_h_dict[step] + np.dot(v_dict[step], self.W_dict[step]))
            prob_h_dict.update({step:val})
        return prob_h_dict

    def sampling_h(self, v_sample_dict):
        prob_h_dict = self.prob_h_dict(v_sample_dict)
        h_sample_dict = {}
        for step in xrange(1, self.rating_num+1):
            h_sample = np.random.binomial(size=prob_h_dict[step].shape, n=1, p=prob_h_dict[step])
            h_sample_dict.update({step:h_sample})
        return prob_h_dict, h_sample_dict

    def sampling_v(self, h_sample_dict):
        prob_v_dict = self.prob_v_dict(h_sample_dict)
        v_sample_dict = {}
        for step in xrange(1,self.rating_num+1):
            v_sample = np.random.binomial(size=prob_v_dict[step].shape, n=1, p=prob_v_dict[step])
            v_sample_dict.update({step:v_sample})
        return prob_v_dict, v_sample_dict

    def gibbs_sampling(self, h_sample_dict):
        prob_v_dict, v_sample_dict = self.sampling_v(h_sample_dict)
        prob_h_dict, h_sample_dict = self.sampling_h(v_sample_dict)
        return prob_v_dict, prob_h_dict, v_sample_dict, h_sample_dict

    def contrastive_divergence(self):
        init_prob_h_dict, init_h_sample_dict = self.sampling_h(self.input)
        for epoch in xrange(self.t):
            if(epoch == 0):
                prob_v_dict, prob_h_dict, v_sample_dict, h_sample_dict = self.gibbs_sampling(init_h_sample_dict)
            else:
                prob_v_dict, prob_h_dict, v_sample_dict, h_sample_dict = self.gibbs_sampling(h_sample_dict)

        for step in xrange(1,self.rating_num+1):
            self.W_dict[step] = self.W_dict[step] + self.learning_rate * (np.dot(self.input[step].T, init_prob_h_dict[step]) - np.dot(v_sample_dict[step].T, prob_h_dict[step]) - self.lmd * self.W_dict[step])
            self.b_v_dict[step] = self.b_v_dict[step] + self.learning_rate * np.mean(self.input[step] - v_sample_dict[step], axis=0)
            self.b_h_dict[step] = self.b_h_dict[step] + self.learning_rate * np.mean(init_prob_h_dict[step] - prob_h_dict[step], axis=0)

    #probability max
    def reconstruct(self, v):
        self.users = v.shape[0]
        conv_v = self.convert_input(v)
        h_dict = self.prob_h_dict(conv_v)
        re_v_dict = self.prob_v_reconstruct(h_dict)
        re_v = np.zeros(v.shape)
        for step in xrange(1, self.rating_num+1):
            re_v += step * re_v_dict[step]
        return re_v

    def convert_input(self, input):
        input_dict = {}
        for step in xrange(1,self.rating_num+1):
            val = np.zeros(input.shape)
            val[np.where(input == step)] = 1
            input_dict.update({step:val})
        return input_dict

    def train(self):
        for epoch in xrange(self.epochs):
            self.contrastive_divergence()

def test_rbm():
    data = np.array([[5, 3, 0, 1],[4, 0, 0, 1],[1, 1, 0, 5],[1, 0, 0, 4],[0, 1, 5, 4]])

    # construct RBM
    rbm = RBM(n_visible=4, n_hidden=2, input=data, epochs=1000)

    # train
    rbm.train()

    # test
    v = np.array([[5, 3, 0, 1],[4, 0, 0, 1],[1, 1, 0, 5],[1, 0, 0, 4],[0, 1, 5, 4]])


    print rbm.reconstruct(v)


if __name__ == "__main__":
    test_rbm()
