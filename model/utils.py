from scipy.spatial.distance import pdist, squareform
import numpy as np
import scipy as sp
from sklearn import datasets

def random_unlabel(true_labels,unlabel_prob=0.1):
    labels = true_labels.copy()
    n = len(labels)
    is_labeled = np.zeros(n)
    is_labeled.fill(True)

    unlabeled_indices = np.random.choice(n, int(n * unlabel_prob), replace=False)
    labeled_indices = np.delete(np.arange(n),unlabeled_indices)
    is_labeled.ravel()[unlabeled_indices] = False
    labels[unlabeled_indices] = 0.5

    return labels, is_labeled, labeled_indices, unlabeled_indices

def rbf_kernel(X,s=None,G=[],percentile=3):
    # use rbf kernel to estimate weights
    pairwise_dists = squareform(pdist(X, 'euclidean'))
    if not s:
        s = 1
    K = sp.exp(-pairwise_dists ** 2 / s ** 2)

    if len(G) == 0:
        threshold = np.percentile(K,percentile)
        np.fill_diagonal(K, 0)

        def get_neighbors(arr):
            # index = arr.argsort()[:-5][::-1]
            index = np.where(arr < threshold)
            arr[index] = 0
            return arr

        K = np.apply_along_axis(get_neighbors, 0, K)
    else:
        K = K * G

    return K

def parallel_coordinates(frame, class_column, cols=None, ax=None, color=None,
                     use_columns=False, xticks=None, colormap=None,
                     **kwds):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    n = len(frame)
    class_col = frame[class_column]
    class_min = np.amin(class_col)
    class_max = np.amax(class_col)

    if cols is None:
        df = frame.drop(class_column, axis=1)
    else:
        df = frame[cols]

    used_legends = set([])

    ncols = len(df.columns)

    # determine values to use for xticks
    if use_columns is True:
        if not np.all(np.isreal(list(df.columns))):
            raise ValueError('Columns must be numeric to be used as xticks')
        x = df.columns
    elif xticks is not None:
        if not np.all(np.isreal(xticks)):
            raise ValueError('xticks specified must be numeric')
        elif len(xticks) != ncols:
            raise ValueError('Length of xticks must match number of columns')
        x = xticks
    else:
        x = range(ncols)

    fig = plt.figure()
    ax = plt.gca()

    Colorm = plt.get_cmap(colormap)

    for i in range(n):
        y = df.iloc[i].values
        kls = class_col.iat[i]
        ax.plot(x, y, color=Colorm((kls - class_min)/(class_max-class_min)), **kwds)

    for i in x:
        ax.axvline(i, linewidth=1, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(df.columns)
    ax.set_xlim(x[0], x[-1])
    ax.legend(loc='upper right')
    ax.grid()

    bounds = np.linspace(class_min,class_max,10)
    cax,_ = mpl.colorbar.make_axes(ax)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=Colorm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%.2f')

    return fig


def accuracy(y,yhat):
    return np.mean((yhat == y))

def rmse(y,yhat):
    return np.mean((yhat - y)**2)

def objective(Ly,Uy_lp,W):
    n = len(Ly) + len(Uy_lp)
    labels = np.hstack((Ly,Uy_lp)).reshape(n,1)
    row, col = np.diag_indices_from(W)
    D = np.identity(W.shape[0])
    D[row,col] = np.sum(W,axis=0)
    return (labels.T @ (D-W) @ labels)[0][0]

def prob_to_one_hot(prob):
    return (prob == prob.max(axis=1)[:,None]).astype(int)

def array_to_one_hot(vec,num_samples, num_classes):
    res = np.zeros((num_samples, num_classes))
    res[np.arange(num_samples), vec.astype(int)] = 1
    return res

def accuracy_mult(sol,pred):
    match = (sol == pred).all(axis=1)
    return np.sum(match) / len(match)
