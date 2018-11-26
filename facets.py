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
from tensorly import unfold, vec_to_tensor, tensor_to_vec
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
        - z0: initial state mean
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
            try:
                (Ev, Evv, Ez, self.U, self.B, self.z0, self.psi0,
                 self.sgm0, self.sgmO, self.sgmR,
                 self.sgmV, self.xi) = _em(
                    self.X, self.W, self.T, self.S,
                    self.L, self.M, self.N, self._lambda,
                    Ev, Evv, self.U, self.B, self.z0, self.psi0,
                    self.sgm0, self.sgmO, self.sgmR, self.sgmV, self.xi)
            except KeyboardInterrupt:
                self.z = Ez
                print(self.z[:3])
                self.save_params()
                break

            self.llh.append(self.compute_log_likelihood())
            # if abs(llh[-1] - llh[-2]) < tol:
            #     print("converged!!")
            #     break
            print("log-likelihood=", self.llh[-1])

            self.z0_log.append(self.z0)
            self.sgm0_log.append(self.sgm0)
            self.sgmO_log.append(self.sgmO)
            self.sgmR_log.append(self.sgmR)
            self.sgmV_log.append(deepcopy(self.sgmV))
            self.xi_log.append(deepcopy(self.xi))

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
        Lm = int(np.prod(L))
        rand_func = np.random.rand
        self.U = [rand_func(N[m], L[m]) for m in range(M)]
        self.B = [rand_func(L[m], L[ m]) for m in range(M)]
        self.z0 = rand_func(Lm) * .001
        self.psi0 = rand_func(Lm, Lm)
        self.sgm0 = rand_func() * 1
        self.sgmO = rand_func() * 1
        self.sgmR = rand_func() * 1
        self.sgmV = rand_func(M) * 1
        self.xi = rand_func(M) * 1

    def _initialize_logger(self):
        self.llh = []
        self.sgm0_log = []
        self.sgmO_log = []
        self.sgmR_log = []
        self.sgmV_log = []
        self.xi_log = []
        self.z0_log = []

    def save_params(self, outdir='./out/tmp/', viz=True):
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.mkdir(outdir)
        np.save(outdir+"X", self.X)
        for m in range(self.M):
            np.savetxt(outdir+f"S_{m}.txt", self.S[m])
            np.savetxt(outdir+f"U_{m}.txt", self.U[m])
            np.savetxt(outdir+f"B_{m}.txt", self.B[m])
        np.save(outdir+"z0.txt", self.z0)
        with open(outdir+"covars.txt", "w") as f:
            f.write(f"sigma_0, {self.sgm0}\n")
            f.write(f"sigma_O, {self.sgmO}\n")
            f.write(f"sigma_R, {self.sgmR}\n")
        np.savetxt(outdir+"sigma_V.txt", self.sgmV)
        np.savetxt(outdir+"Xi.txt", self.xi)
        np.savetxt(outdir+"vec_z.txt", self.z)
        # np.save(outdir+"recon.txt", self.recon_)
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
    plot(facets.z0_log, title="z0",
        xlabel="# of iter (EM)", ylabel="Value",
        outfn=outdir+"z0_log.png")


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
        U, B, z0, psi0, sgm0, sgmO, sgmR, sgmV, xi):
    # compute mode T unfold
    Xt = np.moveaxis(X, -1, 0)
    Wt = np.moveaxis(W, -1, 0)
    for m in range(M):
        print(f"===> mode: {m}")
        """
        Infer the expectations and covariances of
        vectorized latent factors.
        """
        Ez, cov_zzt, cov_zz_, Ezzt, Ezz_ = _e_step(
            Xt, Wt, T, L, N, U, B, z0, psi0, sgm0, sgmO, sgmR, sgmV
        )
        """
        Update parameters
        """
        z0, psi0, sgm0, sgmO, sgmR, sgmV[m], xi[m] = _m_step(
            Xt, Wt, T, S[m], L, N, U, B, z0, sgm0, sgmO, sgmR, sgmV, xi,
            _lambda[m], Ev, Evv, Ez, Ezzt, Ezz_, m
        )

        covZZt = reshape_covariance(cov_zzt, L, m)
        covZZ_ = reshape_covariance(cov_zz_, L, m)
        EZ = reshape_expectation(Ez, L, m)
        print(EZ.shape, covZZt.shape, covZZ_.shape)
        B[m] = update_transition_tensor(
            m, L, B, covZZt, covZZ_, EZ
        )
        # print(B[m])
        U[m] = update_observation_tensor(
            m, Xt, Wt, T, S[m], L, M, N, U, _lambda[m],
            EZ, Ev[m], Evv[m], covZZt,
            z0, sgm0, sgmO, sgmR, sgmV, xi[m]
        )
        # print(U[m])
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

    return Ev, Evv, Ez, U, B, z0, psi0, sgm0, sgmO, sgmR, sgmV, xi

def _e_step(Xt, Wt, T, L, N, U, B, z0, psi0, sgm0, sgmO, sgmR, sgmV):
    # matricize U and B
    matU = kronecker(U, reverse=True)
    matB = kronecker(B, reverse=True)
    # workspace
    Lp = np.prod(L)
    Np = np.prod(N)
    P = np.zeros((T, Lp, Lp))
    J = np.zeros((T, Lp, Lp))
    mu_ = np.zeros((T, Lp))
    psi = np.zeros((T, Lp, Lp))
    mu_h = np.zeros((T, Lp))
    psih = np.zeros((T, Lp, Lp))
    # forward algorithm
    for t in trange(T, desc='forward'):
        ot = tensor_to_vec(Wt[t])  # indices of the observed entries of a tensor X
        xt = tensor_to_vec(Xt[t])[ot]
        lt = sum(ot)  # of observed samples
        Ht = matU[ot, :]
        if t == 0:
            K = sgm0 * Ht.T @ pinv(sgm0 * Ht @ Ht.T + sgmR * np.eye(lt))
            # K = psi0 @ Ht.T @ pinv(Ht @ psi0 @ Ht.T + sgmR * np.eye(lt))
            # psi[0] = sgm0 * np.eye(Lp) - K[0] @ Ht
            psi[0] = sgm0 * (np.eye(Lp) - K[0] @ Ht)
            # psi[0] = (np.eye(Lp) - K @ Ht) @ psi0
            mu_[0] = z0 + K @ (xt - Ht @ z0)
        else:
            P[t-1] = matB @ psi[t-1] @ matB.T + sgmO * np.eye(Lp)
            K = P[t-1] @ Ht.T @ pinv(Ht @ P[t-1] @ Ht.T + sgmR * np.eye(lt))
            mu_[t] = matB @ mu_[t-1] + K @ (xt - Ht @ matB @ mu_[t-1])
            psi[t] = (np.eye(Lp) - K @ Ht) @ P[t-1]

    # backward
    mu_h[-1] = mu_[-1]
    psih[-1] = psi[-1]
    for t in tqdm(list(reversed(range(T-1))), desc='backward'):
        J[t] = psi[t] @ matB.T @ pinv(P[t])
        mu_h[t] = mu_[t] + J[t] @ (mu_h[t+1] - matB @ mu_[t])
        psih[t] = psi[t] + J[t] @ (psih[t+1] - P[t]) @ J[t].T

    # compute expectations
    ztt = np.zeros((T, Lp, Lp))
    zt_ = np.zeros((T, Lp, Lp))
    cov_zt_ = np.zeros((T, Lp, Lp))
    for t in trange(T, desc='compute expectations'):
        if t > 0:
            cov_zt_[t] = psih[t] @ J[t-1].T
            zt_[t] = cov_zt_[t] + np.outer(mu_h[t], mu_h[t-1])
        ztt[t] = psih[t] + np.outer(mu_h[t], mu_h[t])
    zt = mu_h
    cov_ztt = psih
    return zt, cov_ztt, cov_zt_, ztt, zt_

def _m_step(Xt, Wt, T, S, L, N, U, B, z0, sgm0, sgmO, sgmR, sgmV, xi,
            _lambda, Ev, Evv, Ez, Ezzt, Ezz_, mode):
    """
    Eq. (12)
    """
    Umat = kronecker(U, reverse=True)
    Bmat = kronecker(B, reverse=True)
    Lm = np.prod(L)

    z0_new = deepcopy(Ez[0])
    psi0_new = Ezzt[0] - np.outer(Ez[0], Ez[0])
    sgm0_new = np.trace(Ezzt[0] - np.outer(Ez[0], Ez[0])) / Lm
    if sgm0_new < 1.e-7:
        sgm0_new = 1.e-7

    res = np.trace(
        sum(Ezzt[1:])
        - sum(Ezz_[1:]) @ Bmat.T
        - (sum(Ezz_[1:]) @ Bmat.T).T
        + Bmat @ sum(Ezzt[:-1]) @ Bmat.T
    )
    sgmO_new = res / ((T - 1) * Lm)

    res = 0
    for t in range(T):
        Wvec = Wt[t].reshape(-1)
        Xvec = Xt[t].reshape(-1)[Wvec]
        # Wvec = tensor_to_vec(Wt[t])
        # Xvec = tensor_to_vec(Xt[t])[Wvec]
        Uobs = Umat[Wvec, :]
        res += np.trace(Uobs @ Ezzt[t] @ Uobs.T)
        res += Xvec @ Xvec - 2 * Xvec @ (Uobs @ Ez[t])
    print(res, Wt.sum())
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

    return z0_new, psi0_new, sgm0_new, sgmO_new, sgmR_new, sgmV_new, xi_new

def update_observation_tensor(
    mode, Xt, Wt, T, S, L, M, N, U, _lambda,
    EZ, Ev, Evv, cov_ZZt, z0, sgm0, sgmO, sgmR, sgmV, xi):
    """
    Eq. (19), (20)
    """
    G = kronecker(U, skip_matrix=mode, reverse=True).T
    for i in trange(N[mode], desc=f"update U[{mode}]"):
        A_11, A_12, A_21, A_22 = _compute_A(
            _lambda, mode, i, G, Xt, Wt, T, S, L, M, N,
            EZ, Ev, Evv, cov_ZZt
        )
        numer = _lambda * A_11 / xi + (1 - _lambda) * A_12 / sgmR
        denom = _lambda * A_21 / xi + (1 - _lambda) * A_22 / sgmR
        U[mode][i, :] = numer @ pinv(denom)
    return U[mode]

def _compute_A(_lambda, mode, i, G, Xt, Wt, T, S,
               L, M, N, EZ, Ev, Evv, covZZt):
    Nm = N[mode]
    Lm = L[mode]
    Nn = int(np.prod(N) / Nm)
    Ln = int(np.prod(L) / Lm)
    A_11 = A_21 = 0
    A_12 = np.zeros(Lm)
    A_22 = np.zeros((Lm, Lm))
    if _lambda > 0:
        A_11 = sum([S[i, j] * Ev[j] for j in range(Nm)])
        A_21 = sum(Evv)
    if _lambda == 1:
        return A_11, A_12, A_21, A_22
    for t in range(T):
        Xtm = unfold(Xt[t], mode)
        Wtm = unfold(Wt[t], mode)
        for j in range(Nn):
            if Wtm[i, j] == 0:
                continue
            A_12 += Wtm[i, j] * Xtm[i, j] * EZ[t] @ G[:, j]
            A_22 += Wtm[i, j] * (
                _compute_b(G, covZZt[t], j)
                + np.outer(EZ[t] @ G[:, j], EZ[t] @ G[:, j])
            )
    return A_11, A_12, A_21, A_22

def update_transition_tensor(mode, L, B, covZZt, covZZ_, EZ):
    T = len(covZZt)
    F = kronecker(B, skip_matrix=mode, reverse=True).T
    Lm = L[mode]
    Ln = int(np.prod(L) / Lm)
    C1 = np.zeros((Lm, Lm))
    C2 = np.zeros((Lm, Lm))
    for t in trange(1, T, desc=f'update B[{mode}]'):
        for j in range(Ln):
            C1 += _compute_b(F, covZZt[t-1], j)  # t = 1..T-1
            C1 += np.outer(EZ[t-1] @ F[:, j], EZ[t-1] @ F[:, j])
            C2 += _compute_a(F, covZZ_[t], j)  # t = 2..T
            C2 += np.outer(EZ[t, :, j], EZ[t-1] @ F[:, j])
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
    Q = range(Q)
    I = range(I)
    K = range(K)
    for p, i, q, k in itertools.product(P, I, Q, K):
        b[p, q] += F[k, j] * F[i, j] * cov[p, i, q, k]
    return b

def reshape_expectation(z, ranks, mode):
    Lm = ranks[mode]
    Ln = int(np.prod(ranks) / Lm)
    Z = np.zeros((len(z), Lm, Ln))
    # mode-m matricize E[z(t)]
    for t, zt in enumerate(z):
        Z[t] = unfold(zt.reshape(ranks), mode)
    return Z

def reshape_covariance(cov, ranks, mode):
    M = len(ranks)
    Lm = ranks[mode]
    Ln = int(np.prod(ranks) / Lm)
    mat_cov = np.zeros((len(cov), Lm, Ln, Lm, Ln))
    for t, cov_t in enumerate(cov):
        # 1. revert the cov to tensor form
        cov_t = cov_t.reshape((*ranks, *ranks))
        # 2. permute the order of the mode
        cov_t = np.moveaxis(cov_t, mode, 0)
        cov_t = np.moveaxis(cov_t, mode + M, 0 + M)
        # 3. reshape the reordered covariance tensor
        #    by keeping the 1st and (M+1)-th mode fixed
        #    and concatenating data from the 2nd mode
        #    to the M-th mode in to one mode,
        #    and data from the (M+2)-th mode
        #    to the (2M)-th mode into another mode
        new_shape = (
            cov_t.shape[0], np.prod(cov_t.shape[1:M]),
            cov_t.shape[M], np.prod(cov_t.shape[M + 1:2 * M])
        )
        mat_cov[t] = cov_t.reshape(new_shape)
    return mat_cov

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

    geo = [185, 179, 172, 153, 83, 86, 56, 53, 48]
    # settings
    ranks = [3, 4]
    # weights = [1, 1]
    # weights = [.5, .5]
    weights = [0, 0]


    # infer
    facets = Facets(X[geo, :4, :], ranks, weights)
    # X = np.zeros((3, 10, 5, 100))
    # for t in range(100):
    #     if t == 0:
    #         X[:,:,:,0] = np.random.normal(0, .3, (3,10,5))
    #     else:
    #         X[:,:,:, t] = X[:,:,:,t-1] + np.random.normal(0, .3, (3, 10, 5))
    # facets = Facets(X, ranks, weights)

    facets.em(max_iter=20)
    facets.save_params()
