import os
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from tensorly.tenalg import kronecker
from tensorly import fold, unfold, kruskal_to_tensor
# from sktensor import unfold

from dataset import import_tensor, normalize_tensor

class Facets():
    """
        Notations
        ---------
        Input:
        1. a Network of High-order Time Series (Net-HiTs), i.e.,
            - X: an (M+1)-order tensor
            - W: an indicator tensor
            - S: contextual matrices
            - zeta: one-to-one mapping function
        2. dimention of latent factors L
        3. weight of contextual information Lambda

        Output: model parameter set, i.e., 
        ...
    """
    def __init__(self, latent_rank=None,
                 transition_tensor=None, transition_covariance=None,
                 initial_covariance=None,
                 initial_latent_factor=None,  # Z_0
                 ):
        self.latent_rank = latent_rank


    def em(self, X, n_latent_factors=(),
           _lambda=None, n_iter=10):

        self.M, self.N, W, S_, zeta = _parse_input(X)

        self._initialize_parameters()

        if not _lambda or not len(_lambda) == self.M:
            # self._lambda = np.zeros(self.M)
            self._lambda = np.ones(self.M)

        _em(X, W, S_, zeta, n_latent_factors, self._lambda,
            self.M, self.N, self.U_, self.B_, self.Z0,
            self.sgm_o_, self.sgm_0_, self.sgm_r_, self.xi_, self.sgm_v_)


    def sample(self):
        pass


    def _initialize_parameters(self):
        self.U_ = np.array([np.random.randn(self.N[m], self.latent_rank[m])
                          for m in range(self.M)])
        self.B_ = np.array([np.random.randn(self.latent_rank[m], self.latent_rank[m])
                          for m in range(self.M)])
        self.Z0 = np.random.randn(*self.latent_rank)
        """
            simplify the covariances by assuming that
            the each noise is independent and identically distributed (i.i.d.)
        """
        # contextual_covariance
        self.xi_ = np.random.randn(self.M)
        # observation_covariance
        self.sgm_r_ = np.random.rand()
        # transition_covariance
        self.sgm_o_ = np.random.rand()
        # initial_factor_covariance
        self.sgm_0_ = np.random.rand()
        # latent_variable_covariance
        self.sgm_v_ = np.random.rand(self.M) * .1


def _parse_input(X):
    M = np.ndim(X) - 1  # ignore time mode
    N = X.shape[1:]
    W = np.isnan(X)
    S_ = []
    for m in range(M):
        n_dim = N[m]
        X_m = unfold(X, m + 1)
        X_m[np.isnan(X_m)] = 0.
        S_.append(np.corrcoef(X_m))
    zeta = None
    return M, N, W, S_, zeta

def E_vj(S, U, xi, sgm_v):
    # gamma: L(m) * L(m)
    gamma = (np.dot(U.T, U) + xi ** 2 * sgm_v ** -2).T
    vj = np.dot(gamma, np.dot(U.T, S))
    EV = vj
    EVV = gamma + np.dot(vj, vj.T)
    return EV, EVV

def _E_step(X, W, Z0, U, B, sgm_o_, sgm_0_, sgm_r_, xi_, sgm_v_):
    T = len(X)
    K = [None] * T
    P = [None] * T
    J = [None] * T
    mu = [None] * T
    muh = [None] * T
    phi = [None] * T
    phih = [None] * T

    # forward
    for t in range(T):
        print(t)
        o_t = W[t, :]  # indices of the observed entries of a tensor X
        x_t = X[t, o_t]
        H_t = U[o_t, :]

        if t == 0:
            K[t] = sgm_0_ * H_t.T @ np.linalg.inv(sgm_0_ * H_t @ H_t.T + sgm_r_ * np.eye(H_t.shape[0]))
            mu[t] = Z0 + K[t] @ (x_t - H_t @ Z0)
            phi[t] = sgm_0_ * np.eye(K[t].shape[0]) - K[t] @ H_t
            continue

        P[t-1] = B @ phi[t-1] @ B.T + sgm_o_ * np.eye(B.shape[0])
        K[t] = P[t-1] @ H_t.T @ np.linalg.inv(H_t @ P[t-1] @ H_t.T + sgm_r_ * np.eye(H_t.shape[0]))
        mu[t] = B @ mu[t-1] + K[t] @ (x_t - H_t @ B @ mu[t-1])
        phi[t] = (np.eye(P[t-1].shape[0]) - K[t] @ H_t) @ P[t-1]

    # backward
    muh[-1] = mu[-1]  # ?
    phih[-1] = phi[-1]  # ?
    for t in reversed(range(T-1)):
        print(t)
        J[t] = phi[t] @ B.T @ np.linalg.inv(P[t])
        muh[t] = mu[t] + J[t] @ (muh[t+1] - B @ mu[t])
        phih[t] = phi[t] + J[t] @ (phih[t+1] - P[t]) @ J[t].T




def _M_step():
    pass


def _em(X, W, S_, zeta, L, _lambda,
        M, N, U_, B_, Z0, sgm_o_, sgm_0_, sgm_r_, xi_, sgm_v_, n_iter=10):
    """
        compute the expectations of V^(m) before the first iteration.
    """
    X_ = unfold(X, 0)
    W_ = unfold(W, 0)

    Z0 = Z0.flatten()

    for m in range(M):
        X_m = [unfold(X[t], m) for t in range(len(X))]
        W_m = [unfold(W[t], m) for t in range(len(W))]
        if _lambda[m] > 0:
            for j in range(N[m]):
                # infer expectations
                EV, EVV = E_vj(S_[m], U_[m], xi_[m], sgm_v_[m])


    """
        EM algorithm
    """

    U = kronecker(U_)
    B = kronecker(B_)
    for _ in range(n_iter):
        for m in range(M):
            _E_step(X_, W_, Z0, U, B, sgm_o_, sgm_0_, sgm_r_, xi_[m], sgm_v_[m])

            _M_step()

            if _lambda[m] > 0:
                # update
                pass

def reconstruct_matrix(U, Z, mode):
    # Lemma 3.2
    ind = np.ones(len(U), dtype=bool)
    ind[mode] = False
    return np.dot(np.dot(U[mode], unfold(Z, mode)), kronecker(U[ind]).T)

if __name__ == '__main__':

    X, L = import_tensor('./dat/apple/')
    l, t, k = X.shape
    X = normalize_tensor(X)
    X = X.reshape(t, l, k)  # T * N_1 * ... * N_M

    fc = Facets(latent_rank=(10, 5))
    fc.em(X)