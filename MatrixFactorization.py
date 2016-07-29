import numpy as np

def SVD(mat):
    return np.linalg.svd(mat, full_matrices=True)
