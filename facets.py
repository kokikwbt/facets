import os
import time
import itertools
import numpy as np
import numpy.ma as ma
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from tensorly.tenalg import kronecker
from tensorly import fold, unfold, partial_unfold, kruskal_to_tensor
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
    def __init__(self, tensor, latent_rank, contextual_weights):
        self.M = tensor.ndim - 1
        self.N = tensor.shape[:-1]
        indicator_tensor = ~np.isnan(tensor)
        contextual_matrices = _compute_contextual_matrices(tensor, self.M)
        map_func = None

        self.R = {
            'X': tensor,
            'W': indicator_tensor,
            'S': contextual_matrices,
            'zeta': map_func,
            'M': self.M,
            'N': self.N,
        }
        self.L = latent_rank
        self._lambda = contextual_weights

    def em(self, n_iter_max=10):

        self._initialize_parameters()

        # calculate the expectations of V(m)
        _compute_EV(self.R, self._lambda, self.theta)

        # start EM algorithm
        for _ in range(n_iter_max):
            self.theta = _em(self.R, self.M, self.N,
                             self.L, self._lambda, self.theta)

    def sample(self):
        pass


    def _initialize_parameters(self):
        rand_func = np.random.rand

        U_ = np.array([rand_func(self.N[m], self.L[m])
                      for m in range(self.M)])
        # for m in range(self.M):
        #     plt.imshow(U_[m])
        #     plt.show()

        B_ = np.array([rand_func(self.L[m], self.L[m])
                      for m in range(self.M)])
        # for m in range(self.M):
        #     plt.imshow(B_[m])
        #     plt.show()

        Z0 = rand_func(*self.L)

        # contextual_covariance
        xi_ = np.random.randn(self.M)

        # observation_covariance
        sigma_R = np.random.rand()

        # transition_covariance
        sigma_O = np.random.rand()

        # initial_factor_covariance
        sigma_0 = np.random.rand()

        # latent_variable_covariance
        sigma_V_ = np.random.rand(self.M)

        self.theta = {
            "U": U_, "B": B_, "Z0": Z0,
            "sigma_O": sigma_O, "sigma_0": sigma_0, "sigma_R": sigma_R,
            "xi": xi_, "sigma_V": sigma_V_
        }

def _compute_contextual_matrices(tensor, n_modes):
    return [pd.DataFrame(unfold(tensor, m).T).corr().values for m in range(n_modes)]

def _compute_EV(R, _lambda, theta):
    X = R['X']
    W = R['W']
    S = R['S']
    M = R['M']
    N = R['N']
    T = X.shape[-1]
    Xt = np.moveaxis(X, -1, 0)

    U_ = theta["U"]
    xi = theta["xi"]
    sigma_V = theta["sigma_V"]

    Evj = [None] * M
    Evjvj = [None] * M
    for m in range(M):
        Xt_ = np.array([unfold(Xt[t], m) for t in range(T)])
        if _lambda[m] > 0:
            for j in range(N[m]):
                upsilon = sp.linalg.inv(U_[m].T @ U_[m] + xi[m] / sigma_V[m])
                Evj[m] = upsilon @ U_[m].T @ S[m][j, :]
                Evjvj[m] = upsilon + Evj[m] @ Evj[m].T
    return Evj, Evjvj



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
    muh[-1] = mu[-1]  #
    phih[-1] = phi[-1]  #
    for t in reversed(range(T-1)):
        print(t)
        J[t] = phi[t] @ B.T @ np.linalg.inv(P[t])
        muh[t] = mu[t] + J[t] @ (muh[t+1] - B @ mu[t])
        phih[t] = phi[t] + J[t] @ (phih[t+1] - P[t]) @ J[t].T

    E_z = muh
    cov_z = phih
    cov__ = [None] + [phih[t] @ J[t-1].T for t in range(1, T)]
    E_z_ = [None] + [cov__[t] + muh[t] @ muh[t-1].T for t in range(1, T)]
    E_zz = [cov_z[t] + muh[t] @ muh[t-1].T for t in range(T)]

    return E_z, cov_z, cov__, E_z_, E_zz

def _M_step(X_, W_, mode, N, L, S, U_, B_, E_V, E_VV, E_z, cov_z, cov__, E_z_, E_zz, _lambda, xi_, sgm_v_):
    T = len(X_)
    B = kronecker(B_)
    U = kronecker(U_)

    Z0 = E_z[0]

    _L =  np.prod(L)
    sgm_0_ = (np.trace(E_zz[0] - E_z[0] @ E_z[0].T)) / _L

    sgm_o_ = np.trace(np.sum(E_zz[1:]) - B * np.sum(E_z_[1:])
                      - np.sum(E_z_[1:]) * B.T + B * np.sum(E_zz[:-1]) @ B.T
                      ) / (T - 1) * _L
    print('Sigma_O', sgm_o_)

    num = 0
    W_sum = np.sum(W_)
    for t in range(T):
        X_vec = X_[t].flatten()
        W_vec = W_[t].flatten()
        X_vec = X_vec[W_vec]
        U_sub = U[W_vec, :]
        # np.trace(U_sub @ E_zz[t] @ U_sub.T) can be smaller then 0
        num += X_vec.T @ X_vec + np.trace(U_sub @ E_zz[t] @ U_sub.T) - 2 * X_vec.T @ U_sub @ E_z[t]
    sgm_r_ = num / W_sum
    print('Sigma_R', sgm_r_)

    if _lambda > 0:
        U_m = U_[mode]
        # update xi and sigma_V_m
        xi_ = np.sum([S[j, :] @ S[j, :].T - 2 * S[j, :].T @ U_m @ E_V[j]
                    + np.trace(U_m @ E_VV[j] @ U_m.T)
                    for j in range(N[mode])]) / N[mode] ** 2
        print('xi_m:', xi_)

        sgm_v_ = np.sum([np.trace(E_VV[j]) for j in range(N[mode])]) / N[mode] * L[mode] 
        print('Sigma_Vm:', sgm_v_)

    # update B and U

    if _lambda > 0:
        pass

    return Z0, sgm_o_, sgm_0_, sgm_r_, xi_, sgm_v_

def _em(X, W, S_, zeta, L, _lambda,
        M, N, U_, B_, Z0, sgm_o_, sgm_0_, sgm_r_, xi_, sgm_v_, n_iter=10):
    """
        compute the expectations of V^(m) before the first iteration.
    """
    X_ = unfold(X, 0)
    W_ = unfold(W, 0)
    print(np.sum(W_), W_.size)
    Z0 = Z0.flatten()

    E_V = [None] * M
    E_VV = [None] * M
    for m in range(M):
        E_V_ = [None] * N[m]
        E_VV_ = [None] * N[m]
        if _lambda[m] > 0:
            for j in range(N[m]):
                E_V_[j], E_VV_[j] = E_vj(S_[m], U_[m], xi_[m], sgm_v_[m])
        E_V[m], E_VV[m] = E_V_, E_VV_

    """
        EM algorithm
    """

    U = kronecker(U_)
    B = kronecker(B_)
    for _ in range(n_iter):
        for m in range(M):

            E_z, cov_z, cov__, E_z_, E_zz = _E_step(
                X_, W_, Z0, U, B, sgm_o_, sgm_0_, sgm_r_, xi_[m], sgm_v_[m]
            )

            Z0, sgm_o_, sgm_0_, sgm_r_, xi_[m], sgm_v_[m] = _M_step(
                X_, W_, m, N, L, S_[m], U_, B_, E_V[m], E_VV[m],
                E_z, cov_z, cov__, E_z_, E_zz, _lambda[m],
                xi_[m], sgm_v_[m]
            )


def reconstruct_matrix(U, Z, mode):
    # Lemma 3.2
    ind = np.ones(len(U), dtype=bool)
    ind[mode] = False
    return np.dot(np.dot(U[mode], unfold(Z, mode)), kronecker(U[ind]).T)


if __name__ == '__main__':

    X, L = import_tensor('./dat/apple/')
    l, t, k = X.shape
    X = normalize_tensor(X)
    X = np.moveaxis(X, 1, -1)  # N_1 * ... * N_M * T
    print(X.shape)
    L = [10, 5]
    _lambda = np.ones(X.ndim-1)
    # for i in range(l):
    #     plt.plot(X[:,i,:])
    #     plt.show()

    fc = Facets(X, L, _lambda)
    fc.em(X[-10:])