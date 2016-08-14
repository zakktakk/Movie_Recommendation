from __future__ import division
import numpy as np
import pandas as pd
from gensim.models import word2vec

sentences = word2vec.Text8Corpus("timeseries.txt")
model = word2vec.Word2Vec(sentences, size=100, window=5, workers=4, min_count=2)
