import numpy as np
from utils import rbf_kernel
from numpy.linalg import inv


class LP:

    def tnorm(self,X):
        W = rbf_kernel(X)
        T = W / np.sum(W, axis=0)
        Tnorm = T / np.sum(T, axis=1)
        return Tnorm

    def closed(self,X,Ly):
        n = len(Ly)
        Tnorm = self.tnorm(X)
        Tuu_norm = Tnorm[n:,n:]
        Tul_norm = Tnorm[n:,:n]
        Uy_lp = inv((np.identity(len(Tuu_norm))-Tuu_norm)) @ Tul_norm @ Ly
        return Uy_lp

    def iter_(self,X,Ly,Uy,iter_):
        Tnorm = self.tnorm(X).T
        Y = np.hstack((Ly,Uy))

        for i in range(iter_):
            Y = np.dot(Y,Tnorm)
            Y[:len(Ly)] = Ly

        return(Y[len(Ly):])
