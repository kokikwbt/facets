import os
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorly as tl
from sklearn.preprocessing import normalize
from tensorly.tenalg import kronecker
from tensorly import fold, unfold, vec_to_tensor, kruskal_to_tensor
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
           _lambda=None, n_iter=1):

        self.M, self.N, W, S_, zeta = _parse_input(X)

        self._initialize_parameters()

        if not _lambda or not len(_lambda) == self.M:
            # self._lambda = np.zeros(self.M)
            self._lambda = np.ones(self.M)

        _em(X, W, S_, zeta, self.latent_rank, self._lambda,
            self.M, self.N, self.U_, self.B_, self.Z0,
            self.sgm_o_, self.sgm_0_, self.sgm_r_, self.xi_, self.sgm_v_)


    def sample(self):
        pass


    def _initialize_parameters(self):

        # observation tensor
        self.U_ = np.array([np.random.randn(self.N[m], self.latent_rank[m])
                          for m in range(self.M)])
        # multi-linear transition tensor
        self.B_ = [np.random.randn(self.latent_rank[m], self.latent_rank[m])
                          for m in range(self.M)]
        # initial latent factor
        self.Z0 = np.random.rand(*self.latent_rank)

        """
            simplify the covariances by assuming that
            the each noise is independent and identically distributed (i.i.d.)
        """
        # contextual_covariance
        self.xi_ = np.random.randn(self.M)
        # observation_covariance
        self.sgm_r_ = np.random.rand()
        # multi-linear transition covariance
        self.sgm_o_ = np.random.rand()
        # initial_factor_covariance
        self.sgm_0_ = np.random.rand()
        # latent_variable_covariance
        self.sgm_v_ = np.random.rand(self.M)


def _parse_input(X):
    M = np.ndim(X) - 1  # ignore time mode
    N = X.shape[1:]  
    W = ~np.isnan(X)
    S_ = [None] * M
    for m in range(M):
        X_m = unfold(X, m + 1)
        X_m[np.isnan(X_m)] = 0.
        S_[m] = np.corrcoef(X_m)
    zeta = None
    return M, N, W, S_, zeta

def E_vj(S, U, xi, sgm_v):
    gamma = np.linalg.inv(U.T @ U + xi / sgm_v)
    vj = gamma @ U.T @ S
    # print(vj.shape)
    return vj, gamma + vj @ vj.T

def _E_step(X, W, Z0, U, B, sgm_o_, sgm_0_, sgm_r_, xi_, sgm_v_):
    T = len(X)
    K = [None] * T
    P = [None] * T
    J = [None] * T
    mu = [None] * T
    muh = [None] * T
    psi = [None] * T
    psih = [None] * T

    # forward
    for t in range(T):

        o_t = W[t, :]  # indices of the observed entries of a tensor X
        x_t = X[t, o_t]
        H_t = U[o_t, :]

        if t == 0:
            K[t] = sgm_0_ * H_t.T @ np.linalg.inv(sgm_0_ * H_t @ H_t.T + sgm_r_ * np.eye(H_t.shape[0]))
            mu[t] = Z0 + K[t] @ (x_t - H_t @ Z0)
            psi[t] = sgm_0_ * np.eye(K[t].shape[0]) - K[t] @ H_t

        else:
            P[t-1] = B @ psi[t-1] @ B.T + sgm_o_ * np.eye(B.shape[0])
            K[t] = P[t-1] @ H_t.T @ np.linalg.inv(H_t @ P[t-1] @ H_t.T + sgm_r_ * np.eye(H_t.shape[0]))
            mu[t] = B @ mu[t-1] + K[t] @ (x_t - H_t @ B @ mu[t-1])
            psi[t] = (np.eye(P[t-1].shape[0]) - K[t] @ H_t) @ P[t-1]

    # backward
    muh[-1] = mu[-1]  #
    psih[-1] = psi[-1]  #
    for t in reversed(range(T-1)):
        print(t)
        J[t] = psi[t] @ B.T @ np.linalg.inv(P[t])
        muh[t] = mu[t] + J[t] @ (muh[t+1] - B @ mu[t])
        psih[t] = psi[t] + J[t] @ (psih[t+1] - P[t]) @ J[t].T

    E_z = muh
    cov_z = psih
    cov__ = [None] + [psih[t] @ J[t-1].T for t in range(1, T)]
    E_z_ = [None] + [cov__[t] + muh[t] @ muh[t-1].T for t in range(1, T)]
    E_zz = [psih[t] + muh[t] @ muh[t].T for t in range(T)]

    return E_z, cov_z, cov__, E_z_, E_zz

def _M_step(X_, W_, mode, N, L, S, U_, B_, E_V, E_VV, E_z, cov_z, cov__, E_z_, E_zz, _lambda, xi_, sgm_v_):
    T = len(X_)
    B = kronecker(B_[::-1])
    U = kronecker(U_[::-1])

    Z0 = E_z[0]
    _L =  np.prod(L)

    sgm_0_ = (np.trace(E_zz[0] - E_z[0] @ E_z[0].T)) / _L

    sgm_o_ = np.trace(
        np.sum(E_zz[1:])
        - B * np.sum(E_z[1:])  # ?
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
    Z0 = Z0.flatten()
    E_V = [None] * M
    E_VV = [None] * M
    for m in range(M):
        E_V_ = [None] * N[m]
        E_VV_ = [None] * N[m]
        if _lambda[m] > 0:
            for j in range(N[m]):
                E_V_[j], E_VV_[j] = E_vj(S_[m][j, :], U_[m], xi_[m], sgm_v_[m])
                # print(E_V_[j].shape, E_VV_[j].shape)
        E_V[m], E_VV[m] = E_V_, E_VV_

    """
        EM algorithm
    """

    U = kronecker(U_[::-1])
    B = kronecker(B_[::-1])
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

            B_[m] = update_transition_tensor(m, B_, cov_z, cov__, E_z)

            U_[m] = update_observation_tensor(X, W, N, S_[m], E_z, L, U_, E_V[m], E_VV[m], cov_z, m, xi_[m], _lambda[m], sgm_r_)

            if _lambda[m] > 0:
                # update the expectations related to V(m)
                E_V_ = [None] * N[m]
                E_VV_ = [None] * N[m]
                for j in range(N[m]):
                    E_V_[j], E_VV_[j] = E_vj(S_[m][j, :], U_[m], xi_[m], sgm_v_[m])
                E_V[m], E_VV[m] = E_V_, E_VV_

    return U_, Z, V

def update_observation_tensor(X, W, N, S, Z, L, U, E_V, E_VV, cov_Z, mode, xi, _lambda, sgm_R):

    M = len(N)
    G = kronecker([U[m] for m in range(M) if not m == mode][::-1]).T
    # print('G', G.shape)
    cov_Z = reshape_covariance(cov_Z, L, mode)

    for i in range(U[mode].shape[0]):

        A_11, A_12, A_21, A_22 = _compute_A(X, W, N, S[i, :], Z, L, E_V, E_VV, cov_Z, G, mode, i)

        numer = _lambda * A_11 / xi + (1 - _lambda) * A_12 / sgm_R
        denom = _lambda * A_21 / xi + (1 - _lambda) * A_22 / sgm_R

        U[mode][i, :] = numer / denom  # shape -> 10,

    return U[mode]

def _compute_A(X, W, N, S, Z, L, V, VV, cov_Z, G, mode, i):

    T = len(X)
    M = len(N)
    N_n = np.prod([N[m] for m in range(M) if not m == mode])
    L_n = np.prod([L[m] for m in range(M) if not m == mode])

    Z = [vec_to_tensor(Z[t], L) for t in range(T)]

    A_11 = A_12 = A_21 = A_22 = 0

    for j in range(N[mode]):
        A_11 += S[j] * V[j].T

    for j in range(N[mode]):
        # A_21 += VV[j]  # (10,)
        A_21 += V[j] @ V[j].T
        # A_21 += V[j] * V[j].T

        # print("E[VV']", V[j] @ V[j].T)
        # print('VV', V[j].shape)

    for t in range(T):
        Xt = unfold(X[t], mode)
        Wt = unfold(W[t], mode)
        Zt = unfold(Z[t], mode)

        for j in range(N_n):
            # print('Z(t)', Zt.shape)
            A_12 += Wt[i, j] * Xt[i, j] * (Zt @ G[:, j]).T

        # for j in range(L_n):            
            # print(G.T[j,:].shape)
            A_22 += Wt[i, j] * (_compute_b(G[:, j], cov_Z[t]) 
                                + Zt * (G[:, j] @ G[:, j].T) @ Zt.T)

    print('A_11:', A_11.shape)
    print('A_12:', A_12.shape)
    print('A_21:', A_21.shape, A_21)
    print('A_22:', A_22.shape)
    return A_11, A_12, A_21, np.sum(A_22, axis=0)

def update_transition_tensor(mode, B, cov_z, cov_z_, E_z):

    T = len(cov_z)
    M = len(B)
    F = kronecker([B[m] for m in range(M) if not m == mode][::-1]).T
    L = [B[m].shape[0] for m in range(M)]
    L_n = int(np.prod(L) / B[mode].shape[0])
    cov_z = reshape_covariance(cov_z, L, mode)
    cov_z_ = [None] + reshape_covariance(cov_z_[1:], L, mode)
    E_z = reshape_expectation(E_z, L, mode)
    C_1 = C_2 = 0

    for t in range(1, T):
        for j in range(L_n):
            C_1 += _compute_b(F[:, j], cov_z[t - 1])
            C_1 += E_z[t-1] * (F[j] @ F.T[j]) @ E_z[t-1].T
            C_2 += _compute_a(F[:, j], cov_z_[t][:, j, :, :])
            C_2 += E_z[t-1] * F.T[j] @ E_z[t-1].T
    return C_2 / C_1

def _compute_a(F, cov):
    N1, N3, _ = cov.shape
    a = np.zeros((N1, N3))
    for  p in range(N1):
        for q in range(N3):
            for k in range(len(F)):
                a[p, q] += F[k] * cov[p, q, k]
    return a

def _compute_b(F, cov):
    N1, _, N3, _ = cov.shape
    b = np.zeros((N1, N3))
    for p in range(N1):
        for q in range(N3):
            for i, k in itertools.permutations(range(len(F)), 2):
                b[p, q] += F[k] * F[i] * cov[p, i, q, k]
    # print(b.shape)
    return b

def reshape_expectation(E, rank, mode):
    M = len(rank)
    mat_E = [None] * len(E)
    for i, e in enumerate(E):
        e = vec_to_tensor(e, rank)
        e = np.moveaxis(e, mode, 0)
        new_shape = (e.shape[0], np.sum(e.shape[1:]))
        mat_E[i] = e.reshape(new_shape)
    return mat_E

def reshape_covariance(cov, rank, mode):
    M = len(rank)
    mat_cov = [None] * len(cov)
    for i, c in enumerate(cov):
        c = vec_to_tensor(c, (*rank, *rank))
        c = np.moveaxis(c, mode, 0)
        c = np.moveaxis(c, mode + M, M)
        new_shape = (c.shape[0], np.sum(c.shape[1:M]),
                     c.shape[M], np.sum(c.shape[M + 1:2 * M]))
        mat_cov[i] = c.reshape(new_shape)
    return mat_cov

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
    fc.em(X[-10:])