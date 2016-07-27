from __future__ import devision
from scipy.spatial.distance import jaccard, cosine
from scipy.stats import pearsonr

def correlation(d1, d2):
    return pearsonr(d1, d2)[0]

def cosine_similarity(d1, d2):
    return 1 - cosine(d1,d2)

def jaccard_index(d1, d2):
    return 1 - jaccard(d1, d2)

def adjusted_cosine(d1, d2):
    return 
