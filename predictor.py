from __future__ import division
import numpy as np
import pandas as pd
import math
import word2vec
import SlopeOne

user_num = 943
item_num = 1682
rating_num = 100000

#u.data -> user_id, item_id, rating, time_stamp
df_data = pd.read_csv('ml-100k/u.data', sep='\t', header=None)
#u.user -> user_id, age, gender,occupation, zip code
df_user = pd.read_csv('ml-100k/u.user', sep='|', header=None)
#u.item -> movie id | movie title | release date | video release date |IMDb URL | unknown | Action | Adventure | Animation |Children's | Comedy | Crime | Documentary | Drama | Fantasy |Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |Thriller | War | Western |
df_item = pd.read_csv('ml-100k/u.item', sep='|', header=None)

#convert u.user's gender and occupation to each id
#read file and make occupation list
def get_occupation_list(path):
    ret = []
    f = open(path)
    lines = f.readlines()
    f.close()
    for line in lines:
        ret.append(line.strip())
    return ret

#convert string to id
def convert_column(num, lst):
    for (i, element) in enumerate(lst):
        df_user[num][df_user[num] == element] = i

convert_column(2, ['M', 'F'])
convert_column(3, get_occupation_list('ml-100k/u.occupation'))

#apply item2vec using gensim
def get_item_timeseries():
    ret = []
#    for i in range(1, user_num + 1):
    for i in range(1, user_num + 1):
        ret.append(np.asarray(df_data.iloc[df_data[df_data[0] == i].sort(3).index][1], dtype='int32'))
    return np.array(ret)

def get_cosine_matrix(mat):
    size = mat.shape[0]
    ret = np.zeros((size, size))
    for i in xrange(size):
        for j in xrange(size):
            if i == j:
                break
            else:
                val = cosine_similarity(mat[i], mat[j])
                ret[i][j] = val
                ret[j][i] = val
    return ret

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / math.sqrt(np.dot(vec1, vec1) * np.dot(vec2, vec2))
cosine = get_cosine_matrix(word2vec.CBoW(get_item_timeseries()))
print np.max(cosine)
print np.argmax(cosine)

"""
#make user-item-evaluate matrix
def get_eval_matrix():
    ret = np.zeros((user_num, item_num))
    for i, data in df_data.iterrows():
        ret[data[0] - 1][data[1] - 1] = data[2]
    return ret

print SlopeOne.bipolar_slope_one(get_eval_matrix())
"""
