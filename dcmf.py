import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from scipy.linalg import pinv

class DCMF(object):
    """
    Dynamic Contextual Matrix Factorization
    """
    def __init__(self, X, rank, weight=None):
        if not X.ndim > 1:
            raise ValueError("")
        self.T, self.n = T, n = X.shape
        self.l = l = rank
        self.X = X
        self.W = ~np.isnan(X)
        self.S = np.random.rand(n, n)
        if not weight:
            self.lmd = 1.
        # init params
        self.U = np.random.rand(n, l)
        self.B = np.random.rand(l, l)
        self.z0 = np.random.rand(l)
        self.psi0 = np.random.rand(l, l)
        self.sgmZ = np.random.rand()
        self.sgmX = np.random.rand()
        self.sgmS = np.random.rand()
        self.sgmV = np.random.rand()

    def em(self, max_iter=100):
        for iteration in range(max_iter):
            forward(self.l, self.X, self.W, self.U,
                    self.z0, self.psi0, self.sgmX)
            backward()
            for j in range(N):
                pass
        self.Z = None
        self.V = None
        self.recon_ = self.U @ self.Z

    def save_model(self):
        pass

def forward(l, X, W, U, z0, psi0, sgmX):
    """
    ot: the indices of the observed entriesof xt
    Ht: the corresponding compressed version of U

    x^*t = Ht @ zt + Gaussian noise
    """
    T, n = X.shape
    mu_ = np.zeros((T, l))
    psi = np.zeros((T, l, l))
    K = np.zeros((T, l, n))
    for t in range(T):
        ot = W[t, :]
        lt = sum(ot)
        x = X[t, ot]
        H = U[ot, :]
        # construct Ht based on Eq. 3.6
        if t == 0:
            K[0] = psi0 @ H.T @ pinv(H @ psi0 @ H.T + sgmX * np.eye(n))
            psi[0] = (np.eye(l) - K[0] @ H) @ psi0
            mu_[0] = z0 + K[0] @ (x - H @ z0)
        else:
            pass
        # Estimate mu and phi
    return

def backward():
    pass


if __name__ == '__main__':

    X = np.loadtxt('./dat/86_11.amc.4d', delimiter=',')
    X = scale(X)
    print(X.shape)

    model = DCMF(X, 2)
    model.em()
