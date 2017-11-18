import numpy as np
from utils import rbf_kernel
from numpy.linalg import inv


class LP:
    def __init__(self,num_nodes):
        self.num_nodes = num_nodes

    def tnorm(self,W):
        T = W / np.sum(W, axis=0, keepdims=True)
        Tnorm = T / np.sum(T, axis=1, keepdims=True)
        return Tnorm

    def closed(self,W,Ly):
        n = len(Ly)
        Tnorm = self.tnorm(W)
        Tuu_norm = Tnorm[n:,n:]
        Tul_norm = Tnorm[n:,:n]
        Uy_lp = inv((np.identity(len(Tuu_norm))-Tuu_norm)) @ Tul_norm @ Ly
        return Uy_lp

    def iter_(self,W,Ly,Uy,iter_):
        Tnorm = self.tnorm(W)
        Y = np.hstack((Ly,Uy))

        for i in range(iter_):
            Y = np.dot(Y,Tnorm)
            Y[:len(Ly)] = Ly

        return(Y[len(Ly):])
