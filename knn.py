from __future__ import division
import numpy as np
import math
import sys

def euclidean_distanve(obj, feature):
    return np.sum((obj - feature) ** 2)

def cosine_distance(obj, feature):
    cosine_similarity = np.dot(obj, feature) / math.sqrt(np.dot(obj, obj) * np.dot(featurem, feature))
    if cosine_similarity < 1 / sys.maxint:
        return sys.maxint
    return 1 / cosine_similarity

def get_k_nearest(k, distance):
    dist_list = np.ones(k) * sys.maxint
    ret = np.zeros(k)
    for i, dist in enumerate(distance):
        if max(ret) > dist:
            dist_list[np.argmax(ret)] = dist
            ret[np.argmax(ret)] = i
    return ret

#get mini distmce
def k_nearest_neighbor(k, feature_mat, obj_vec):
    distance_vac = np.zeros(len(feature_mat))
    for i, feature in enumerate(feature_mat):
        distance_vac[i] =  cosine_distance(obj_vec, feature)
    ret = get_k_nearest(k, distance_vac)
    return　ret
