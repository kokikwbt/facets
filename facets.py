import os
import shutil
import time
import itertools
import warnings
from copy import deepcopy
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.linalg import pinv
from tensorly import unfold, vec_to_tensor
from tensorly.tenalg import kronecker
from tensorly.tucker_tensor import tucker_to_tensor
from tqdm import tqdm, trange
from dataset import import_tensor, normalize_tensor
from myplot import *
warnings.filterwarnings("ignore")


class Facets(object):
    """
    Notations
    =========
    Input:
        1. a Network of High-order Time Series (Net-HiTs)
            - X: an (M+1)-order tensor
            - W: an indicator tensor
            - S: contextual matrices
            - zeta: one-to-one mapping function
        2. L: dimention of latent factors
        3. Lambda: weight of contextual information
    Output: model parameter set:
        - U: observation tensor
        - B: transition tensor
        - Z0: initial state mean
        - sgm0: initial state covariance
        - sgmO: transition covariance
        - sgmR: observation covariance
        - sgmV(m): latent context covariance for m-th mode
        - xi(m): context covariance
    """
    def __init__(self, tensor, rank, weights):
        self.X = tensor
        self.W = ~np.isnan(tensor)
        self.T = T = tensor.shape[-1]
        self.L = L = rank
        self.M = M = tensor.ndim - 1
        self.N = N = tensor.shape[:-1]
        self.S = compute_contextual_matrices(tensor, M)
        self._lambda = weights  # contextual weights
        self.zeta = None  # map function (?)
        self._initialize_parameters()
        self._initialize_logger()
        print("\n\nINPUT STATUS")
        print("=====================================")
        print(f"- input tensor: {N} * {T}")
        print(f"- # of missing values: {np.prod(N)*T-self.W.sum()}/{np.prod(N)*T}")
        print("- objective rank:", L)
        print("- contextual weights:", self._lambda)
        print("=====================================\n")

    def em(self, max_iter=10, tol=1.e-7, init=False):
        if init: self._initialize_parameters()
        # initialize the expectations of V(m)
        Ev, Evv, = _compute_context_expectation(
            self.L, self.M, self.N, self.S, self.U,
            self.xi, self.sgmV, self._lambda
        )
        # start EM algorithm
        for iter in range(max_iter):
            print('================')
            print(' iter', iter + 1)
            print('================')
            (Ev, Evv, Ez, self.U, self.B, self.Z0,
             self.sgm0, self.sgmO, self.sgmR, self.sgmV, self.xi) = _em(
                self.X, self.W, self.T, self.S, self.L, self.M, self.N,
                self._lambda, Ev, Evv, self.U, self.B,
                self.Z0, self.sgm0, self.sgmO, self.sgmR, self.sgmV, self.xi
            )

            self.llh.append(self.compute_log_likelihood())
            # if abs(llh[-1] - llh[-2]) < tol:
            #     print("converged!!")
            #     break
            print("log-likelihood=", self.llh[-1])

            sgmV = deepcopy(self.sgmV)
            xi = deepcopy(self.xi)
            self.sgm0_log.append(self.sgm0)
            self.sgmO_log.append(self.sgmO)
            self.sgmR_log.append(self.sgmR)
            self.sgmV_log.append(sgmV)
            self.xi_log.append(xi)

        # EZ = Ez.reshape((self.T, *self.L))
        # print(EZ.shape, self.U[0].shape)
        self.z = Ez
        self.recon_ = recon_ = np.array([
            tucker_to_tensor(Ez[t].reshape(*self.L), self.U)
            for t in range(self.T)
        ])
        print(self.recon_.shape)
        self.recon_ = np.moveaxis(recon_, -1, 1)
        # for i in range(self.recon_.shape[1]):
            # plt.plot(self.recon_[:, i, :])
            # plt.show()

    def compute_log_likelihood(self):
        llh = 0
        # C1. tensor time series
        # C2. contextual information
        # C3. temporal smoothness
        return llh

    def sample(self, n_samples):
        pass

    def _initialize_parameters(self, setseed=False):
        if setseed: np.random.seed(777)
        L, M, N = self.L, self.M, self.N
        rand_func = np.random.rand
        self.U = [rand_func(N[m], L[m]) * 2 - 1 for m in range(M)]
        self.B = [rand_func(L[m], L[m]) * 2 - 1 for m in range(M)]
        self.Z0 = rand_func(*self.L) * 2 - 1
        self.sgm0 = rand_func() * 5
        self.sgmO = rand_func() * 5
        self.sgmR = rand_func() * 5
        self.sgmV = rand_func(M) * 2
        self.xi = rand_func(M) * 2

    def _initialize_logger(self):
        self.llh = []
        self.sgm0_log = []
        self.sgmO_log = []
        self.sgmR_log = []
        self.sgmV_log = []
        self.xi_log = []

    def save_params(self, outdir='./out/tmp/', viz=True):
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.mkdir(outdir)
        np.save(outdir+"X", self.X)
        for m in range(self.M):
            np.savetxt(outdir+f"S_{m}.txt", self.S[m])
            np.savetxt(outdir+f"U_{m}.txt", self.U[m])
            np.savetxt(outdir+f"B_{m}.txt", self.B[m])
        np.save(outdir+"Z0.txt", self.Z0)
        with open(outdir+"covars.txt", "w") as f:
            f.write(f"sigma_0, {self.sgm0}\n")
            f.write(f"sigma_O, {self.sgmO}\n")
            f.write(f"sigma_R, {self.sgmR}\n")
        np.savetxt(outdir+"sigma_V.txt", self.sgmV)
        np.savetxt(outdir+"Xi.txt", self.xi)
        np.savetxt(outdir+"vec_z.txt", self.z)
        np.save(outdir+"recon.txt", self.recon_)
        if viz: viz_params(self, outdir)

def viz_params(facets, outdir):
    for m in range(facets.M):
        heatmap(facets.S[m], title=f"S_{m}",
                xlabel="dimension", ylabel="dimension",
                outfn=outdir+f"S_{m}.png")
        heatmap(facets.U[m], title=f"observation tensor U_{m}",
                xlabel="dim. of latent factor",
                ylabel="dim. of observations",
                outfn=outdir+f"U_{m}.png")
        heatmap(facets.B[m], title=f"transition tensor B_{m}",
                xlabel="dim. of latent factor",
                ylabel="dim. of latent factor",
                outfn=outdir+f"B_{m}.png")
    plot(facets.sgm0_log, title="sigma_0",
        xlabel="# of iter (EM)", ylabel="Value",
        outfn=outdir+"sigma_0_log.png")
    plot(facets.sgmO_log, title="sigma_O",
        xlabel="# of iter (EM)", ylabel="Value",
        outfn=outdir+"sigma_O_log.png")
    plot(facets.sgmR_log, title="sigma_R",
        xlabel="# of iter (EM)", ylabel="Value",
        outfn=outdir+"sigma_R_log.png")
    plot(facets.sgmV_log, title="sigma_V",
        xlabel="# of iter (EM)", ylabel="Value",
        outfn=outdir+"sigma_V_log.png")
    plot(facets.xi_log, title="Xi",
        xlabel="# of iter (EM)", ylabel="Value",
        outfn=outdir+"xi_log.png")


def compute_contextual_matrices(tensor, n_modes):
    return [pd.DataFrame(unfold(tensor, m).T).corr().values
            for m in range(n_modes)]

def _context_expectation_aux(N, L, S, U, xi, sgmV):
    Ev = np.zeros((N, L))
    Evv = np.zeros((N, L, L))
    for j in range(N):
        # original
        # upsilon = pinv(U.T @ U + xi / sgmV)
        # Ev[j] = upsilon @ U.T @ S[:, j]
        # Evv[j] = upsilon + np.outer(Ev[j], Ev[j])
        # DCMF[5]
        Minv = pinv(U.T @ U + xi / sgmV * np.eye(L))
        upsilon = xi * Minv
        Ev[j] = Minv @ U.T @ S[:, j]
        Evv[j] = upsilon + np.outer(Ev[j], Ev[j])
        # ???
        # Minv = pinv(U.T @ U + xi / sgmV * np.eye(L))
        # upsilon = Minv
        # Ev[j] = Minv @ U.T @ S[:, j]
        # Evv[j] = upsilon + np.outer(Ev[j], Ev[j])
    return Ev, Evv

def _compute_context_expectation(L, M, N, S, U, xi, sgmV, _lambda):
    Ev, Evv = [None] * M, [None] * M
    for m in range(M):
        if _lambda[m] == 0:
            continue
        Ev[m], Evv[m] = _context_expectation_aux(
            N[m], L[m], S[m], U[m], xi[m], sgmV[m]
        )
    return Ev, Evv

def _em(X, W, T, S, L, M, N, _lambda, Ev, Evv,
        U, B, Z0, sgm0, sgmO, sgmR, sgmV, xi):
    # compute mode T unfold
    Xt = unfold(np.moveaxis(X, -1, 0), 0)
    Wt = unfold(np.moveaxis(W, -1, 0), 0)
    for m in range(M):
        print(f"===> mode: {m}")
        """
        Infer the expectations and covariances of
        vectorized latent factors.
        """
        Ez, cov_zzt, cov_zz_, Ezzt, Ezz_ = _e_step(
            Xt, Wt, T, L, N, U, B, Z0, sgm0, sgmO, sgmR, sgmV
        )
        """
        Update parameters
        """
        Z0, sgm0, sgmO, sgmR, sgmV[m], xi[m] = _m_step(
            Xt, Wt, T, S[m], L, N, U, B, Z0, sgm0, sgmO, sgmR, sgmV, xi,
            _lambda[m], Ev, Evv, Ez, Ezzt, Ezz_, m
        )

        cov_ZZt = reshape_covariance(cov_zzt, L, m)
        cov_ZZ_ = reshape_covariance(cov_zz_[1:], L, m)
        EZ = reshape_expectation(Ez, L, m)
        B[m] = update_transition_tensor(
            m, B, L, cov_ZZt, cov_ZZ_, EZ
        )
        U[m] = update_observation_tensor(
            m, X, W, T, S[m], L, M, N, U, _lambda[m],
            EZ, Ev[m], Evv[m], cov_ZZt,
            Z0, sgm0, sgmO, sgmR, sgmV, xi[m]
        )

        if _lambda[m] > 0:
            # update the expectations related to V(m)
            Ev[m], Evv[m] = _context_expectation_aux(
                N[m], L[m], S[m], U[m], xi[m], sgmV[m]
            )

    print("===> sgm0:", sgm0)
    print("===> sgmO:", sgmO)
    print("===> sgmR:", sgmR)
    print("===> sgmV:", sgmV)
    print("===> xi:", xi, "\n")

    return Ev, Evv, Ez, U, B, Z0, sgm0, sgmO, sgmR, sgmV, xi

def _e_step(Xt, Wt, T, L, N, U, B, Z0, sgm0, sgmO, sgmR, sgmV):
    # matricize U and B
    U = kronecker(U[::-1])
    B = kronecker(B[::-1])
    # vectorize Z0
    z0 = Z0.reshape(-1)
    # workspace
    Lp = np.prod(L)
    Np = np.prod(N)
    K = [None] * T  # (T, Lp, ?), "?" depends on # of observations
    P = np.zeros((T, Lp, Lp))
    J = np.zeros((T, Lp, Lp))
    mu_ = np.zeros((T, Lp))
    psi = np.zeros((T, Lp, Lp))
    mu_h = np.zeros((T, Lp))
    psih = np.zeros((T, Lp, Lp))

    # forward
    for t in trange(T, desc='forward'):
        ot = Wt[t, :]  # indices of the observed entries of a tensor X
        lt = sum(ot)
        xt = Xt[t, ot]
        Ht = U[ot, :]
        if t == 0:
            K[0] = sgm0 * Ht.T @ pinv(sgm0 * Ht @ Ht.T + sgmR * np.eye(lt))
            # K[0] = sgm0 * Ht.T @ pinv(Ht @ (sgm0 * np.eye(Lp)) @ Ht.T + sgmR * np.eye(lt))
            # K[0] = sgm0 * np.eye(Lp) @ Ht.T @ pinv(Ht @ (sgm0 * np.eye(Lp)) @ Ht.T + sgmR * np.eye(lt))
            mu_[0] = z0 + K[0] @ (xt - Ht @ z0)
            # psi[0] = sgm0 * np.eye(Lp) - K[0] @ Ht
            psi[0] = sgm0 * (np.eye(Lp) - K[0] @ Ht)
            # psi[0] = (np.eye(Lp) - K[0] @ Ht) @ (sgm0 * np.eye(Lp))
        else:
            P[t-1] = B @ psi[t-1] @ B.T + sgmO * np.eye(Lp)
            K[t] = P[t-1] @ Ht.T @ pinv(Ht @ P[t-1] @ Ht.T + sgmR * np.eye(lt))
            mu_[t] = B @ mu_[t-1] + K[t] @ (xt - Ht @ B @ mu_[t-1])
            psi[t] = (np.eye(Lp) - K[t] @ Ht) @ P[t-1]

    # backward
    mu_h[-1] = mu_[-1]
    psih[-1] = psi[-1]
    for t in tqdm(list(reversed(range(T-1))), desc='backward'):
        J[t] = psi[t] @ B.T @ pinv(P[t])
        mu_h[t] = mu_[t] + J[t] @ (mu_h[t+1] - B @ mu_[t])
        psih[t] = psi[t] + J[t] @ (psih[t+1] - P[t]) @ J[t].T

    # compute expectations
    Ez = mu_h
    cov_zzt = psih
    cov_zz_ = np.zeros((T, Lp, Lp))
    Ezzt = np.zeros((T, Lp, Lp))
    Ezz_ = np.zeros((T, Lp, Lp))
    for t in trange(T, desc='compute expectations'):
        if t > 0:
            cov_zz_[t] = psih[t] @ J[t-1].T
            Ezz_[t] = cov_zz_[t] + np.outer(mu_h[t], mu_h[t-1])
        Ezzt[t] = psih[t] + np.outer(mu_h[t], mu_h[t])

    return Ez, cov_zzt, cov_zz_, Ezzt, Ezz_

def _m_step(Xt, Wt, T, S, L, N, U, B, Z0, sgm0, sgmO, sgmR, sgmV, xi,
            _lambda, Ev, Evv, Ez, Ezzt, Ezz_, mode):
    """
    Eq. (12)
    """
    Umat = kronecker(U[::-1])
    Bmat = kronecker(B[::-1])
    Lp = np.prod(L)

    Z0_new = vec_to_tensor(Ez[0], L)
    sgm0_new = np.trace(Ezzt[0] - np.outer(Ez[0], Ez[0])) / Lp

    res = sum(Ezz_[1:]) @ Bmat.T
    sgmO_new = np.trace(
        sum(Ezzt[1:]) - res.T - res
        + Bmat @ sum(Ezzt[:-1]) @ Bmat.T
    ) / ((T - 1) * Lp)

    res = 0
    for t in range(T):
        Wvec = Wt[t]
        Xvec = Xt[t, Wvec]
        Uobs = Umat[Wvec, :]
        res += np.trace(Uobs @ Ezzt[t] @ Uobs.T)
        res += Xvec @ Xvec - 2 * Xvec @ (Uobs @ Ez[t])
    sgmR_new = res / Wt.sum()

    if _lambda > 0:
        sgmV_new = sum([np.trace(Evv[mode][j]) for j in range(N[mode])])
        sgmV_new /= (N[mode] * L[mode])
        xi_new = sum([S[:, j] @ S[:, j] - 2 * S[:, j] @ U[mode] @ Ev[mode][j]
                      for j in range(N[mode])])
        xi_new += np.trace(U[mode] @ sum(Evv[mode]) @ U[mode].T)
        xi_new /= N[mode] ** 2
    else:
        sgmV_new = sgmV[mode]
        xi_new = xi[mode]

    return Z0_new, sgm0_new, sgmO_new, sgmR_new, sgmV_new, xi_new

def update_observation_tensor(
    mode, X, W, T, S, L, M, N, U, _lambda,
    EZ, Ev, Evv, cov_ZZt, Z0, sgm0, sgmO, sgmR, sgmV, xi):
    """
    Eq. (19), (20)
    """
    Xt = np.moveaxis(X, -1, 0)
    Wt = np.moveaxis(W, -1, 0)
    G = kronecker([U[m] for m in reversed(range(M)) if not m == mode]).T
    for i in trange(N[mode], desc=f"update U[{mode}]"):
        A_11, A_12, A_21, A_22 = _compute_A(
            _lambda, mode, i, G, Xt, Wt, T, S, L, M, N,
            EZ, Ev, Evv, cov_ZZt
        )
        numer = _lambda * A_11 / xi + (1 - _lambda) * A_12 / sgmR
        denom = _lambda * A_21 / xi + (1 - _lambda) * A_22 / sgmR
        U[mode][i, :] = numer @ pinv(denom)
        row = numer @ pinv(denom)
        # print(col, col.std())
    print(U[mode])
    return U[mode]

def _compute_A(_lambda, mode, i, G, Xt, Wt, T, S,
               L, M, N, EZ, Ev, Evv, cov_ZZt):
    Np = int(np.prod(N) / N[mode])
    Lp = int(np.prod(L) / L[mode])
    A_11 = sum([S[i, j] * Ev[j] for j in range(N[mode])]) if _lambda > 0 else 0
    A_21 = sum(Evv) if _lambda > 0 else 0
    A_12 = A_22 = 0
    for t in range(T):
        Xtm = unfold(Xt[t], mode)
        Wtm = unfold(Wt[t], mode)
        Xtm[~Wtm] = 0  # avoid nan
        A_12 += sum([
            # Wtm[i, j] * Xtm[i, j] * (G[:, j] @ EZ[t].T)  # ?
            Wtm[i, j] * Xtm[i, j] * EZ[t] @ G[:, j]  # ?
            for j in range(Np)
        ])
        # for j in range(Np):
            # heatmap(_compute_b(G, cov_ZZt[t], j))
            # heatmap(EZ[t] @ np.outer(G[:, j], G[:, j]) @ EZ[t].T)
        # plt.show()
        A_22 += sum([
            Wtm[i, j] * (
                # _compute_b(G, cov_ZZt[t], j)
                + EZ[t] @ np.outer(G[:, j], G[:, j]) @ EZ[t].T
            )
            # for j in range(Lp)
            for j in range(Np)
        ])
    return A_11, A_12, A_21, A_22

def update_transition_tensor(mode, B, L, covZZt, covZZ_, EZ):
    M = len(B)
    T = len(covZZt)
    F = kronecker([B[m] for m in reversed(range(M))
                        if not m == mode]).T
    Lm = int(np.prod(L) / L[mode])
    C1 = C2 = 0
    for t in trange(1, T, desc=f'update B[{mode}]'):
        for j in range(Lm):
            # C1 += _compute_b(F, covZZt[t-1], j)  # t = 1..T-1
            C1 += EZ[t-1] @ np.outer(F[:, j], F[:, j]) @ EZ[t-1].T
            # C2 += _compute_a(F, covZZ_[t-1], j)  # t = 2..T
            C2 += np.outer(EZ[t, :, j], F[:, j]) @ EZ[t-1].T
            # C2 += EZ[t, :, j] @ F[:, j] @ EZ[t-1].T
    # print(C1.shape, C2.shape, EZ[0].T.shape)
    return C2 @ pinv(C1)

def _compute_a(F, cov, j):
    P, _, Q, K = cov.shape
    a = np.zeros((P, Q))
    P = range(P)
    Q = range(Q)
    K = range(K)
    for p, q, k in itertools.product(P, Q, K):
        a[p, q] += F[k, j] * cov[p, j, q, k]
    return a

def _compute_b(F, cov, j):
    P, I, Q, K = cov.shape
    b = np.zeros((P, Q))
    P = range(P)
    I = range(I)
    Q = range(Q)
    K = range(K)
    # print(b.shape, F.shape, cov.shape)
    for p, i, q, k in itertools.product(P, I, Q, K):
        b[p, q] += F[k, j] * F[i, j] * cov[p, i, q, k]
    return b

def reshape_expectation(Ez, rank, mode):
    L0 = rank[mode]
    L1 = int(np.prod(rank) / L0)
    EZ = np.zeros((len(Ez), L0, L1))  # matricized E[z(t)]
    for t, zt in enumerate(Ez):
        EZ[t] = unfold(vec_to_tensor(zt, rank), mode)
    return EZ

def reshape_covariance(cov, rank, mode):
    T = len(cov)
    M = len(rank)
    L0 = rank[mode]
    L1 = int(np.prod(rank) / L0)
    covZZ = np.zeros((T, L0, L1, L0, L1))
    for t, cov_t in enumerate(cov):
        # 1. revert the cov to tensor form
        cov_t = vec_to_tensor(cov_t, (*rank, *rank))
        # 2. permute the order of the mode
        cov_t = np.moveaxis(cov_t, mode, 0)
        cov_t = np.moveaxis(cov_t, mode + M, M)
        # 3. reshape the reordered covariance tensor
        #    by keeping the 1st and (M+1)-th mode fixed
        #    and concatenating data from the 2nd mode
        #    to the M-th mode in to one mode,
        #    and data from the (M+2)-th mode
        #    to the (2M)-th mode into another mode
        new_shape = (
            cov_t.shape[0], np.prod(cov_t.shape[1:M]),
            cov_t.shape[M], np.prod(cov_t.shape[M+1:2*M])
        )
        covZZ[t] = cov_t.reshape(new_shape)
    return covZZ

def reconstruct_matrix(U, Z, mode):
    # Lemma 3.2
    index = np.ones(len(U), dtype=bool)
    index[mode] = False
    return U[mode] @ Z @ kronecker(U[ind][::-1]).T


if __name__ == '__main__':

    # load dataset
    X, countries = import_tensor('./dat/apple/')
    # X = normalize_tensor(X)
    Y = X.flatten()
    Y = Y - np.nanmean(Y)
    Y /= np.nanstd(Y)
    print(np.nanmean(Y), np.nanstd(Y))
    X = Y.reshape(X.shape)
    X = np.moveaxis(X, 1, -1)  # N_1 * ... * N_M * T
    for i, geo in enumerate(countries):
        print(i, geo.name)

    geo = [185, 179, 172, 153, 86, 83, 56, 53, 48]
    # settings
    ranks = [4, 3]
    # weights = [1, 1]
    weights = [1, 1]

    # infer
    facets = Facets(X[geo[:5], :, -300:], ranks, weights)
    facets.em(max_iter=20)
    facets.save_params()
