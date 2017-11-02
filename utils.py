from scipy.spatial.distance import pdist, squareform
import numpy as np
import scipy as sp
from sklearn import datasets

def get_iris_data(label_prob=0.05):
    """ Read the iris data and label/unlabel data points"""
    # load iris data
    iris   = datasets.load_iris()
    data   = iris["data"]
    labels = iris["target"]

    # get label 0 and 1, and corresponding data
    labels = labels[labels < 2]
    data = data[np.where(labels < 2)]

    # generate random numbers for unlabeling
    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(len(labels)) < 1-label_prob
    masked_labels = np.copy(labels).astype(float)

    # keep labels for points that will be unlabeled
    Uy_sol = np.copy(labels[random_unlabeled_points])

    # unlabel points (opposite of the true label, to make the problem hard)
    masked_labels[random_unlabeled_points] = 1-labels[random_unlabeled_points]

    unlabeled_indices = np.where(random_unlabeled_points)

    # separate labeled/unlabeled Y
    Uy = masked_labels[random_unlabeled_points]
    Ly = np.delete(masked_labels,unlabeled_indices)

    # separate labeled/unlabeled X
    UX = data[unlabeled_indices]
    LX = np.delete(data,unlabeled_indices,axis=0)

    return LX, Ly, UX, Uy, Uy_sol

def rbf_kernel(X,s=None):
    # use rbf kernel to estimate weights
    pairwise_dists = squareform(pdist(X, 'euclidean'))
    if not s:
        s = np.var(X)
    K = sp.exp(-pairwise_dists ** 2 / s ** 2)
    threshold = np.percentile(K,10)
    np.fill_diagonal(K, 0)

    def get_neighbors(arr):
        # index = arr.argsort()[:-5][::-1]
        index = np.where(arr < threshold)
        arr[index] = 0
        return arr

    K = np.apply_along_axis(get_neighbors, 0, K)

    return K

def accuracy(y,yhat):
    return np.mean((yhat == y))

def rmse(y,yhat):
    return np.mean((yhat - y)**2)

def objective(Ly,Uy_lp,W):
    labels = np.hstack((Ly,Uy_lp)).reshape(100,1)
    return (labels.T @ W @ labels)[0][0]
