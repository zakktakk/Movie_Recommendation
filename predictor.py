import numpy as np
import pandas as pd
import RBM
import MatrixFactorization
import SlopeOne
import evaluate

user_num = 943
item_num = 1682
rating_num = 100000

#u.data -> user_id, item_id, rating, time_stamp
df_base = pd.read_csv('ml-100k/ua.base', sep='\t', header=None)
#u.data -> user_id, item_id, rating, time_stamp
df_test = pd.read_csv('ml-100k/ua.test', sep='\t', header=None)

#u.user -> user_id, age, gender,occupation, zip code
df_user = pd.read_csv('ml-100k/u.user', sep='|', header=None)
#u.item -> movie id | movie title | release date | video release date |IMDb URL | unknown | Action | Adventure | Animation |Children's | Comedy | Crime | Documentary | Drama | Fantasy |Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |Thriller | War | Western |
df_item = pd.read_csv('ml-100k/u.item', sep='|', header=None)

#make user-item-evaluate matrix
def get_eval_matrix():
    ret = np.zeros((user_num, item_num))
    for i, data in df_base.iterrows():
        ret[data[0] - 1][data[1] - 1] = data[2]
    return ret

print SlopeOne.bipolar_slope_one(get_eval_matrix())
