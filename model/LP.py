import numpy as np
from numpy.linalg import inv

import sys
sys.path.append('../')
from model.utils import rbf_kernel

class LP:
    '''
    Label propagation for predicting labels for unlabeled nodes.
    Closed form and iterated solutions.
    See mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf for details.
    '''
    def __init__(self):
        return

    def tnorm(self,weights):
        '''
        Column normalize -> row normalize weights.
        '''
        # column normalize weights
        T = weights / np.sum(weights, axis=0, keepdims=True)
        # row normalize T
        Tnorm = T / np.sum(T, axis=1, keepdims=True)
        return Tnorm

    def closed(self,labels,
                    weights,
                    labeled_indices,
                    unlabeled_indices):
        '''
        Closed solution of label propagation.
        '''
        # normalize T
        Tnorm = self.tnorm(weights)
        # sort Tnorm by unlabeled/labeld
        Tuu_norm = Tnorm[np.ix_(unlabeled_indices,unlabeled_indices)]
        Tul_norm = Tnorm[np.ix_(unlabeled_indices,labeled_indices)]
        # closed form prediction for unlabeled nodes
        lapliacian = (np.identity(len(Tuu_norm))-Tuu_norm)
        propagated = Tul_norm @ labels[labeled_indices]
        label_predictions = np.linalg.solve(lapliacian, propagated)
        return label_predictions

    def iter_(self,labels,
                   weights,
                   labeled_indices,
                   unlabeled_indices,
                   num_iter):
        '''
        Iterated solution of label propagation.
        '''
        # normalize T
        Tnorm = self.tnorm(weights)
        Y = labels.copy()

        for i in range(num_iter):
            # propagate labels
            Y = np.dot(Tnorm,Y)
            # don't update labeled nodes
            Y[labeled_indices] = labels[labeled_indices]

        # only return label predictions
        return(Y[unlabeled_indices])

    def iter_multiclass(self,W,Ly,num_classes,num_unlabeled,iter_=-1):
        preds = []
        for class_ in range(num_classes):
            Ly_class = Ly == class_
            Uy_class = np.array([1/num_classes] * num_unlabeled)
            if iter_ == -1:
                pred = self.closed(W,Ly_class)
            else:
                pred = self.iter_(W,Ly_class,Uy_class,iter_)
            preds.append(pred)
        res = np.vstack(preds).T
        return res
