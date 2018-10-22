import os
import time
import itertools
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tensorly as tl
import pprint
from scipy import linalg
# from numpy import linalg
from sklearn.preprocessing import normalize
from tensorly.tenalg import kronecker
from tensorly import fold, unfold, vec_to_tensor, kruskal_to_tensor
from tqdm import trange
from dataset import import_tensor, normalize_tensor

pp = pprint.PrettyPrinter(indent=2)

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
        Ev, Evv, = _compute_expect_context(self.R, self._lambda, self.theta)

        # start EM algorithm
        for _ in range(n_iter_max):
            self.theta, Ev, Evv = _em(
                self.R, self.L, self._lambda, self.theta, Ev, Evv
            )
            pp.pprint(self.theta)

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

def _expect_context_aux(N, S, U, xi, sigma_V):
    Evm = [None] * N
    Evvm = [None] * N
    for j in range(N):
        upsilon = linalg.pinv(U.T @ U + xi / sigma_V)
        Evm[j] = upsilon @ U.T @ S[j, :]
        Evvm[j] = upsilon + Evm[j] @ Evm[j].T
    return Evm, Evvm

def _compute_expect_context(R, _lambda, theta):
    S, M, N = R["S"], R["M"], R["N"]
    U_ = theta["U"]
    xi = theta["xi"]
    sigma_V = theta["sigma_V"]
    Ev = [None] * M
    Evv = [None] * M
    for m in range(M):
        if not _lambda[m] > 0:
            continue
        Ev[m], Evv[m] = _expect_context_aux(N[m], S[m], U_[m], xi[m], sigma_V[m])
    return Ev, Evv

def _em(R, L, _lambda, theta, Ev, Evv):
    """
    Main loop
    """
    X, W, S, M, N = R['X'], R["W"], R["S"], R["M"], R["N"]
    T = X.shape[-1]
    Xt = np.moveaxis(X, -1, 0)  # time-mode unfold
    Wt = np.moveaxis(W, -1, 0)

    B = theta["B"]
    U = theta["U"]
    xi = theta["xi"]
    sigma_V = theta["sigma_V"]

    for m in range(M):
        """
        Infer the expectations and covariances of
        vectorized latent factors.
        """
        Ez, covzz_, Ezz_, covzz, Ezz, = _estep(unfold(Xt, 0), unfold(Wt, 0), R, theta)

        """
        Update parameters
        """
        theta = _mstep(Xt, Wt, R, L, m, _lambda[m], theta, Ev, Evv, Ez, Ezz_, Ezz)
        covZZ_ = reshape_covariance(covzz_[1:], L, m)
        covZZ = reshape_covariance(covzz, L, m)
        EZm = reshape_expectation(Ez, L, m)
        B[m] = update_transition_tensor(m, B, L, covZZ, covZZ_, EZm)
        U[m] = update_observation_tensor(m, R, L, _lambda[m], EZm, Ev[m], Evv[m], covZZ, theta)
        # plt.imshow(B[m]); plt.colorbar(), plt.show()
        # plt.imshow(U[m]); plt.colorbar(), plt.show()

        if _lambda[m] > 0:
            # update the expectations related to V(m)
            Ev[m], Evv[m] = _expect_context_aux(N[m], S[m], U[m], xi[m], sigma_V[m])

    # update theta
    theta["U"] = U
    theta["B"] = B

    return theta, Ev, Evv

def _estep(Xt, Wt, R, theta):
    print('Xt:', Xt.shape)
    U = kronecker(theta["U"][::-1])
    B = kronecker(theta["B"][::-1])
    print('U:', U.shape, 'B', B.shape)

    z0 = theta["Z0"].reshape(-1)
    sigma_R = theta["sigma_R"]
    sigma_O = theta["sigma_O"]
    sigma_0 = theta["sigma_0"]
    sigma_V = theta["sigma_V"]

    T = len(Xt)
    K = [None] * T
    P = [None] * T
    J = [None] * T
    mu = [None] * T
    mu_hat = [None] * T
    psi = [None] * T
    psi_hat = [None] * T

    # forward
    for t in trange(T, desc='forward'):
        # print(f't = {t}')
        # print('---> # of observations:', Wt[t].sum())
        ot = Wt[t, :]  # indices of the observed entries of a tensor X
        xt = Xt[t, ot]
        Ht = U[ot, :]
        # print('---> Ht', Ht.shape)

        if t == 0:
            K[0] = sigma_0 * Ht.T @ linalg.inv(sigma_0 * Ht @ Ht.T + sigma_R * np.eye(Ht.shape[0]))
            mu[0] = z0 + K[0] @ (xt - Ht @ z0)
            psi[0] = sigma_0 * np.eye(K[0].shape[0]) - K[0] @ Ht
            # pp.pprint(K[0])
            # pp.pprint(mu[0])
            # pp.pprint(psi[0])

        else:
            P[t-1] = B @ psi[t-1] @ B.T + sigma_O * np.eye(B.shape[0])
            K[t] = P[t-1] @ Ht.T @ linalg.inv(Ht @ P[t-1] @ Ht.T + sigma_R * np.eye(Ht.shape[0]))
            mu[t] = B @ mu[t-1] + K[t] @ (xt - Ht @ B @ mu[t-1])
            psi[t] = (np.eye(K[t].shape[0]) - K[t] @ Ht) @ P[t-1]
            # pp.pprint(P[t-1])
            # pp.pprint(K[t])
            # pp.pprint(mu[t])
            # pp.pprint(psi[t])

    # backward
    mu_hat[-1] = mu[-1]  #
    psi_hat[-1] = psi[-1]  #
    for t in reversed(range(T-1)):
        # print(f't = {t}')
        J[t] = psi[t] @ B.T @ linalg.inv(P[t])
        mu_hat[t] = mu[t] + J[t] @ (mu_hat[t+1] - B @ mu[t])
        psi_hat[t] = psi[t] + J[t] @ (psi_hat[t+1] - P[t]) @ J[t].T

    Ez = mu_hat
    covzz_ = [None] + [psi_hat[t] @ J[t-1].T for t in range(1, T)]
    Ezz_ = [None] + [covzz_[t] + mu_hat[t] @ mu_hat[t-1].T for t in range(1, T)]
    covzz = psi_hat
    Ezz = [psi_hat[t] + mu_hat[t] @ mu_hat[t].T for t in range(T)]

    # pp.pprint(Ez)
    # pp.pprint(covzz_)
    # pp.pprint(Ezz_)
    # pp.pprint(covzz)
    # pp.pprint(Ezz)

    return Ez, covzz_, Ezz_, covzz, Ezz

def _mstep(Xt, Wt, R, L, m, _lambda, theta, Evj, Evjvj, Ez, Ezz_, Ezz):
    M = R["M"]
    N = R["N"]
    Sm = R["S"][m]
    T = R["X"].shape[-1]
    U = kronecker(theta["U"][::-1])
    B = kronecker(theta["B"][::-1])
    U_ = theta["U"]
    Um = theta["U"][m]
    """
    Equation (12)
    """
    vec_Z0_new = Ez[0]

    sigma_0_new = np.trace(Ezz[0] - Ez[0] * Ez[0].T) / np.prod(L)  # Ez[0].T ?
    sigma_O_new = np.trace(
        sum(Ezz[1:])
        - B @ sum(Ezz_[1:])  # ?
        - sum(Ezz_[1:] @ B.T)
        + B * sum(Ezz[:-1] @ B.T)
    ) / ((T - 1) * np.prod(L))

    val = 0
    for t in range(T):
        vec_Wt = Wt[t].reshape(-1)
        vec_Xt = Xt[t].reshape(-1)[vec_Wt]
        mat_U_obs = U[vec_Wt, :]
        val += (vec_Xt.T @ vec_Xt
                + np.trace(mat_U_obs @ Ezz[t] @ mat_U_obs.T)
                - 2 * vec_Xt.T @ mat_U_obs @ Ez[t])

    sigma_R_new = val / Wt.sum()

    theta["Z0"] = vec_Z0_new
    theta["sigma_O"] = sigma_O_new
    theta["sigma_0"] = sigma_0_new
    theta["sigma_R"] = sigma_R_new

    if _lambda > 0:
        sigma_Vm_new = sum([np.trace(Evjvj[m][j]) for j in range(N[m])]) / (N[m] * L[m])

        xim_new = sum([Sm[j, :].T @ Sm[j, :] - 2 * Sm.T[j, :] @ Um @ Evj[m][j]
                      + np.trace(Um @ Evjvj[m][j] @ Um.T)
                      for j in range(N[m])]
                      ) / N[m] ** 2
        theta["xi"][m] = xim_new
        theta["sigma_V"][m] = sigma_Vm_new

    return theta

def update_observation_tensor(mode, R, L, _lambda, EZm, Evm, Evvm, covZZ, theta):
    M = R["M"]
    U = theta["U"]
    xi = theta["xi"][mode]
    sigma_R = theta["sigma_R"]
    G = kronecker([U[m] for m in range(M) if not m == mode][::-1]).T

    for i in trange(U[mode].shape[0], desc=f'update U[{mode}]'):
        A_11, A_12, A_21, A_22 = _compute_A(R, L, EZm, Evm, Evvm, covZZ, G, mode, i)
        numer = _lambda * A_11 / xi + (1 - _lambda) * A_12 / sigma_R
        denom = _lambda * A_21 / xi + (1 - _lambda) * A_22 / sigma_R
        # plt.imshow(linalg.pinv(denom))
        # plt.show()
        U[mode][i, :] = numer @ linalg.pinv(denom)  # shape -> 10,

    return U[mode]

def _compute_A(R, L, EZm, Evm, Evvm, covZZ, G, mode, i):
    X = R["X"]
    W = R["W"]
    S = R["S"][mode]
    M = R["M"]
    N = R["N"]
    Xt = np.moveaxis(X, -1, 0)
    Wt = np.moveaxis(W, -1, 0)
    T = X.shape[-1]
    Nprod = np.prod([N[m] for m in range(M) if not m == mode])
    Lprod = np.prod([L[m] for m in range(M) if not m == mode])

    A_11 = sum([S[i, j] * Evm[j].T for j in range(N[mode])])

    A_21 = sum(Evvm)

    A_12 = A_22 = 0

    for t in range(T):
        Xtm = unfold(Xt[t], mode)
        Wtm = unfold(Wt[t], mode)
        Xtm[np.isnan(Xtm)] = 0.

        A_12 += sum(
            [Wtm[i, j] * Xtm[i, j] * (EZm[t] @ G[:, j]).T
            for j in range(Nprod)]
        )  # nan??

        A_22 += sum(
            [Wtm[i, j] * (_compute_b(G[:, j], covZZ[t])
            + EZm[t] * (G[:, j] @ G[:, j].T) @ EZm[t].T)
            for j in range(Lprod)]
        )

    # print('A_11:', A_11.shape)
    # print('A_12:', A_12.shape)
    # print('A_21:', A_21.shape)
    # print('A_22:', A_22.shape)

    return A_11, A_12, A_21, A_22

def update_transition_tensor(mode, B, L, covZZ, covZZ_, EZ):
    T = len(covZZ)
    M = len(B)
    F = kronecker([B[m] for m in range(M) if not m == mode][::-1]).T
    Ln = int(np.prod(L) / B[mode].shape[0])
    C_1 = C_2 = 0
    for t in range(1, T):
        for j in range(Ln):
            C_1 += _compute_b(F[:, j], covZZ[t-2])
            C_1 += EZ[t-1] * (F[j] @ F.T[j]) @ EZ[t-1].T
            C_2 += _compute_a(F[:, j], covZZ_[t-1][:, j, :, :])
            C_2 += EZ[t-1] * F.T[j] @ EZ[t-1].T
    return C_2 / C_1

def _compute_a(F, cov):
    N1, N3, _ = cov.shape
    a = np.zeros((N1, N3))
    for  p in range(N1):
        for q in range(N3):
            for k in range(len(F)):
                a[p, q] += F[k] * cov[p, q, k]
    return a

def _compute_b(X, cov):
    N1, _, N3, _ = cov.shape
    b = np.zeros((N1, N3))
    for p in range(N1):
        for q in range(N3):
            for i, k in itertools.permutations(range(len(X)), 2):
                b[p, q] += X[k] * X[i] * cov[p, i, q, k]
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
    X = np.moveaxis(X, 1, -1)  # N_1 * ... * N_M * T
    print(X.shape)

    L = [10, 5]
    _lambda = np.ones(X.ndim-1) * 1

    facets = Facets(X[:, :, -50:], L, _lambda)
    facets.em()