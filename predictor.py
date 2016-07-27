import numpy as np
import pandas as pd
import word2vec

user_num = 943
item_num = 1682
rating_num = 100000

#u.data -> user_id, item_id, rating, time_stamp
df_data = pd.read_csv('ml-100k/u.data', sep='\t', header=None)
#u.user -> user_id, age, gender, occupation, zip code
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
    for i in range(1, 5):
        ret.append(np.asarray(df_data.iloc[df_data[df_data[0] == i].sort(3).index][1], dtype='int32'))
    return ret

word2vec.CBoW(get_item_timeseries())
