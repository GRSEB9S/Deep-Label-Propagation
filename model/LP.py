import numpy as np
from numpy.linalg import inv

import sys
sys.path.append('../')
from model.utils import rbf_kernel


class LP:
    def __init__(self,num_nodes,num_labeled):
        self.num_nodes   = num_nodes
        self.num_labeled = num_labeled

    def t(self,W):
        return W / np.sum(W, axis=0, keepdims=True)

    def tnorm(self,W):
        T = self.t(W)
        Tnorm = T / np.sum(T, axis=1, keepdims=True)
        return Tnorm

    def closed(self,W,Ly):
        n = len(Ly)
        Tnorm = self.tnorm(W)
        Tuu_norm = Tnorm[n:,n:]
        Tul_norm = Tnorm[n:,:n]
        a = (np.identity(len(Tuu_norm))-Tuu_norm)
        b = Tul_norm @ Ly
        Uy_lp = np.linalg.solve(a, b)
        return Uy_lp

    def iter_(self,W,Ly,Uy,iter_):
        Tnorm = self.tnorm(W)
        Y = np.hstack((Ly,Uy))

        for i in range(iter_):
            Y = np.dot(Y,Tnorm)
            Y[:self.num_labeled] = Ly

        return(Y[self.num_labeled:])

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
