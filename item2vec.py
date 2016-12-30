#-*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
from gensim.models import word2vec

class Item2Vec:
    def __init__(self, fname):
        self.make_model(fname)
    
    def make_model(self, fname):
        data = word2vec.Text8Corpus(fname)
        self.model = word2vec.Word2Vec(data)

    def get_topk(self, movie, k):
        result = np.array(self.model.most_similar(positive=[movie], topn=k))
        return result
