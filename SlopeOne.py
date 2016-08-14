from __future__ import division
import numpy as np
#http://www.tani.cs.chs.nihon-u.ac.jp/g-2009/mori/sotuken/mori_re.pdf

#slope one
def slope_one(eval_mat):
    user_num = eval_mat.shape[0]
    item_num = eval_mat.shape[1]
    #get average deviation
    def get_dev_val(i, j):
        dev_val = 0
        users = 0
        for row in xrange(user_num):
            #if the user evaluated both movie i and movie j
            if((eval_mat[row][i] != 0) and (eval_mat[row][j] != 0)):
                users += 1
                dev_val += eval_mat[row][i] - eval_mat[row][j]
        #avoid zero division
        if(users == 0):
            return 0
        return dev_val / users
    #get average diviation
    dev = np.zeros((item_num, item_num))
    for i in xrange(item_num):
        for j in xrange(item_num):
            if i == j:
                break
            else:
                dev_temp = get_dev_val(i, j)
                dev[i][j] = dev_temp
                dev[j][i] = (-1) * dev_temp
    #get predictive evaluation matrix
    pred_mat = np.zeros((user_num, item_num))
    for u in xrange(user_num):
        print u
        eval_row = np.where(eval_mat[u] != 0)[0]
        for j in xrange(item_num):
            pred_mat[u][j] = (np.sum(dev[j][eval_row] + eval_mat[u][eval_row])) / len(eval_row)
    return pred_mat

#weighted slope one
def weighted_slope_one(eval_mat):
    user_num = eval_mat.shape[0]
    item_num = eval_mat.shape[1]
    #get average deviation
    def get_dev_val(i, j):
        dev_val = 0
        users = 0
        for row in xrange(user_num):
            #if the user evaluated both movie i and movie j
            if((eval_mat[row][i] != 0) and (eval_mat[row][j] != 0)):
                users += 1
                dev_val += eval_mat[row][i] - eval_mat[row][j]
        #avoid zero division
        if(users == 0):
            ret = 0
        else:
            ret = dev_val / users
        return ret, users
    #get average diviation
    dev = np.zeros((item_num, item_num))
    evaled_users_mat = np.zeros((item_num, item_num))
    for i in xrange(item_num):
        for j in xrange(item_num):
            if i == j:
                break
            else:
                dev_temp, users = get_dev_val(i, j)
                dev[i][j] = dev_temp
                dev[j][i] = (-1) * dev_temp
                evaled_users_mat[i][j] = users
                evaled_users_mat[j][i] = users
    #get predictive evaluation matrix
    pred_mat = np.zeros((user_num, item_num))
    for u in xrange(user_num):
        print u
        eval_row = np.where(eval_mat[u] != 0)[0]
        for j in xrange(item_num):
            pred_mat[u][j] = np.sum((dev[j][eval_row] + eval_mat[u][eval_row]) * evaled_users_mat[j][eval_row]) / np.sum(evaled_users_mat[j][eval_row])
    return pred_mat

#bi-polar slope one
def bipolar_slope_one(eval_mat):
    user_num = eval_mat.shape[0]
    item_num = eval_mat.shape[1]

    def average_evaluation(eval_mat):
        ret = np.mean(eval_mat, axis = 1)
        items = eval_mat.shape[1]
        for row in xrange(ret.shape[0]):
            nonzero = np.count_nonzero(eval_mat[row])
            ret[row] *= items / nonzero
        return ret

    ave_eval_lst = average_evaluation(eval_mat)
    print ave_eval_lst

    #get average deviation
    def get_dev_val_like(i, j):
        dev_val = 0
        users_like = 0
        for row in xrange(user_num):
            #if the user evaluated both movie i and movie j
            threshold = ave_eval_lst[row]
            if((eval_mat[row][i] > threshold) and (eval_mat[row][j] > threshold)):
                users_like += 1
                dev_val += eval_mat[row][i] - eval_mat[row][j]
        #avoid zero division
        if(users_like == 0):
            ret = 0
        else:
            ret = dev_val / users_like
        return ret, users_like

    #get average deviation
    def get_dev_val_dislike(i, j):
        dev_val = 0
        users_dislike = 0
        for row in xrange(user_num):
            #if the user evaluated both movie i and movie j
            threshold = ave_eval_lst[row]
            if((0 < eval_mat[row][i] < threshold) and (0 < eval_mat[row][j] < threshold)):
                users_dislike += 1
                dev_val += eval_mat[row][i] - eval_mat[row][j]
        #avoid zero division
        if(users_dislike == 0):
            ret = 0
        else:
            ret = dev_val / users_dislike
        return ret, users_dislike

    #get average diviation
    dev_like = np.zeros((item_num, item_num))
    dev_dislike = np.zeros((item_num, item_num))
    evaled_like_users_mat = np.zeros((item_num, item_num))
    evaled_dislike_users_mat = np.zeros((item_num, item_num))

    for i in xrange(item_num):
        for j in xrange(item_num):
            if i == j:
                break
            else:
                dev_like_temp, users_like = get_dev_val_like(i, j)
                dev_like[i][j] = dev_like_temp
                dev_like[j][i] = (-1) * dev_like_temp
                evaled_like_users_mat[i][j] = users_like
                evaled_like_users_mat[j][i] = users_like
                dev_dislike_temp, users_dislike = get_dev_val_dislike(i, j)
                dev_dislike[i][j] = dev_dislike_temp
                dev_dislike[j][i] = (-1) * dev_dislike_temp
                evaled_dislike_users_mat[i][j] = users_dislike
                evaled_dislike_users_mat[j][i] = users_dislike

    #get predictive evaluation matrix
    pred_mat = np.zeros((user_num, item_num))
    for u in xrange(user_num):
        eval_like_row = np.where(eval_mat[u] > ave_eval_lst[u])[0]
        eval_dislike_row = np.where((eval_mat[u] < ave_eval_lst[u]) & (eval_mat[u] > 0))[0]
        for j in xrange(item_num):
            den = np.sum(evaled_like_users_mat[j][eval_like_row]) + np.sum(evaled_dislike_users_mat[j][eval_dislike_row])
            if den != 0:
                nume = np.sum((dev_like[j][eval_like_row] + eval_mat[u][eval_like_row]) * evaled_like_users_mat[j][eval_like_row]) + np.sum((dev_dislike[j][eval_dislike_row] + eval_mat[u][eval_dislike_row]) * evaled_dislike_users_mat[j][eval_dislike_row])
                print nume
                pred_mat[u][j] = nume / den
    return pred_mat

if __name__ == '__main__':
    R = np.array([[4,5,2,4,0,5],[2,0,3,4,3,0],[1,4,0,5,3,4],[0,5,0,0,2,4],[0,3,1,3,0,3]])
    # R = np.array([[5, 3, 0, 1],[4, 0, 0, 1],[1, 1, 0, 5],[1, 0, 0, 4],[0, 1, 5, 4]])
    nR = bipolar_slope_one(R)
    print R
    print nR
