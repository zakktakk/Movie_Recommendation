import numpy as np
from scipy import stats, spatial

class CollaborativeFiltering:
    item_list = []
    user_list = []
    user_item_relation = np.array([[]])
    item_similarity = np.array([[]])
    user_similarity = np.array([[]])

    """
    @param items item name list (np.array)
    @param users user name list (np.array)
    @param evaluation_matrix user item evaluation matrix (np.array)
    @param item_sim item-item similarity matrix (np.array)
    """
    def __init__(self, items, users, evaluation_matrix, item_sim):
        self.item_list = items
        self.user_list = users
        self.user_item_matrix = evaluation_matrix
        self.item_similarity = item_sim
        self.user_similarity = PearsonSim()

    #calvulate user similarity with peason correlation
    def PearsonSim(self):
        usrln = len(self.user_list)
        ret = np.zeros((usrln, usrln))
        for i in xrange(usrln):
            if(i <= j):
                for j in xrange(usrln):
                    tmp = stats.pearsonr(user_item_relation[i], user_item_relation[j])
                    ret[i][j] = tmp
                    ret[j][i] = tmp
        return ret

    #calvulate user similarity with cosine similarity
    def CosineSim(self):
        usrln = len(self.user_list)
        ret = np.zeros((usrln, usrln))
        for i in xrange(usrln):
            if(i <= j):
                for j in xrange(usrln):
                    tmp = 1 - spatial.distance.cosine(user_item_relation[i], user_item_relation[j])
                    ret[i][j] = tmp
                    ret[j][i] = tmp
        return ret

    #calculate predict score using item based CF
    def ItemBasedCF(self):
        usrln = len(self.user_list)
        itmln = len(self.item_list)
        denominator_lst = np.sum(self.item_similarity, axis=0)
        numerator_mat = np.dot(self.user_item_matrix, self.item_similarity)
        ret = np.zeros((usrln, itmln))
        for i in xrange(itmln):
            denomi = denominator_lst[i]
            if(denomi != 0):
                for j in xrange(usrln):
                    ret[j][i] = numerator_mat[j][i]/denomi
        return ret


    #calculate predict score using user based CF
    def UserBasedCF(self):
        usrln = len(self.user_list)
        itmln = len(self.item_list)
        colmn_ave = np.sum(self.user_item_matrix, axis=1)/itmln
        ret = np.zeros((usrln, itmln))
        for i in xrange(usrln):
            for j in xrange(itmln):
                ret = self.user_item_matrix[i][j] - colmn_ave[i]
        denominator_lst = np.sum(self.user_similarity, axis=1)
        numerator_mat = np.dot(self.user_similarity, self.ret)
        for i in xrange(usrln):
            denomi = denominator_lst[i]
            if(denomi != 0):
                for j in xrange(itmln):
                    ret[i][j] = colmn_ave[i] + numerator_mat[i][j]/denomi
        return ret
