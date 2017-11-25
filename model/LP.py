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
        a = (1.00000000001 * np.identity(len(Tuu_norm))-Tuu_norm)
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
