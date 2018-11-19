import os
import shutil
import time
import itertools
import warnings
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.linalg import pinv
from tensorly import fold, unfold, vec_to_tensor, kruskal_to_tensor
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
        print(f"- # of missing values: {np.sum(self.W)}/{np.prod(N)*T}")
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
            print("log-likelihood", self.llh[-1])
            self.sgm0_log.append(self.sgm0)
            self.sgmO_log.append(self.sgmO)
            self.sgmR_log.append(self.sgmR)
            self.sgmV_log.append(self.sgmV)
            self.xi_log.append(self.xi)

        # EZ = Ez.reshape((self.T, *self.L))
        # print(EZ.shape, self.U[0].shape)
        self.recon_ = recon_ = np.array([
            tucker_to_tensor(Ez[t].reshape(*self.L), self.U)
            for t in range(self.T)
        ])
        print(self.recon_.shape)
        for i in range(self.recon_.shape[1]):
            plt.plot(self.recon_[:, i, :])
            plt.show()

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
        self.U = [rand_func(N[m], L[m]) for m in range(M)]
        self.B = [rand_func(L[m], L[m]) for m in range(M)]
        self.Z0 = rand_func(*self.L)
        self.sgm0 = rand_func()
        self.sgmO = rand_func()
        self.sgmR = rand_func()
        self.sgmV = rand_func(M)
        self.xi = rand_func(M)

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
        np.savetxt(outdir+"V.txt", self.sgmV)
        np.savetxt(outdir+"Xi.txt", self.xi)
        np.save(outdir+"recon.txt", self.recon_)
        if viz: viz_params(self)

def viz_params(facets, outdir):
    # plot(facets.X)
    for m in range(M):
        heatmap(facets.S[m], title=f"S_{m}",
                xlabel="dimension", ylabel="dimension",
                outfn=outdir+f"S_{m}.png")
        heatmap(facets.U[m], title=f"observation tensor U_{m}",
                xlabel="dim. of observations",
                ylabel="dim. of latent factor",
                outfn=outdir+f"U_{m}.png")
        heatmap(facets.B[m], title=f"transition tensor B_{m}",
                xlabel="dim. of latent factor",
                ylabel="dim. of latent factor",
                outfn=outdir+f"B_{m}.png")

def compute_contextual_matrices(tensor, n_modes):
    return [pd.DataFrame(unfold(tensor, m).T).corr().values
            for m in range(n_modes)]

def _context_expectation_aux(N, L, S, U, xi, sgmV):
    Ev = np.zeros((N, L))
    Evv = np.zeros((N, L, L))
    for j in range(N):
        upsilon = xi * pinv(U.T @ U + xi / sgmV * np.eye(L))
        Ev[j] = upsilon @ U.T @ S[:,j]
        Evv[j] = upsilon + np.outer(Ev[j], Ev[j])
    return Ev, Evv

def _compute_context_expectation(L, M, N, S, U, xi, sgmV, _lambda):
    Ev, Evv = [None] * M, [None] * M
    for m in range(M):
        if _lambda[m] <= 0:
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
            mu_[0] = z0 + K[0] @ (xt - Ht @ z0)
            # psi[0] = sgm0 * (np.eye(Lp) - K[0] @ Ht)
            psi[0] = sgm0 * np.eye(Lp) - K[0] @ Ht
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

    Z0_new = Ez[0].reshape(L)
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
        res += (
            np.trace(Uobs @ Ezzt[t] @ Uobs.T)
            + Xvec @ Xvec - 2 * Xvec @ (Uobs @ Ez[t])
        )
    sgmR_new = res / Wt.sum()

    if _lambda > 0:
        sgmV_new = sum([np.trace(Evv[mode][j]) for j in range(N[mode])])
        sgmV_new /= (N[mode] * L[mode])
        xi_new = sum([
            S[j, :] @ S[j, :] - 2 * S[j, :] @ U[mode] @ Ev[mode][j]
            + np.trace(U[mode] @ Evv[mode][j] @ U[mode].T)
            for j in range(N[mode])
        ]) / N[mode] ** 2
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

    return U[mode]

def _compute_A(_lambda, mode, i, G, Xt, Wt, T, S, L, M, N, EZ, Ev, Evv, cov_ZZt):
    Np = int(np.prod(N) / N[mode])
    Lp = int(np.prod(L) / L[mode])
    A_11 = sum([S[i, j] * Ev[j].T for j in range(N[mode])]) if _lambda > 0 else 0
    A_21 = sum(Evv) if _lambda > 0 else 0
    A_12 = A_22 = 0
    for t in range(T):
        Xtm = unfold(Xt, mode)
        Wtm = unfold(Wt, mode)
        Xtm[np.isnan(Xtm)] = 0.
        A_12 += sum([
            Wtm[j, i] * Xtm[j, i] * (EZ[t] @ G[:, j]).T
            for j in range(Np)
        ])
        A_22 += sum([
            Wtm[j, i] * _compute_b(G, cov_ZZt[t], j)
            + Wtm[j, i] * EZ[t] @ np.outer(G[:, j], G[:, j]) @ EZ[t].T
            for j in range(Lp)
        ])
    return A_11, A_12, A_21, A_22

def update_transition_tensor(mode, B, L, cov_ZZt, cov_ZZ_, EZ):
    M = len(B)
    T = len(cov_ZZt)
    F = kronecker([B[m] for m in range(M) if not m == mode][::-1]).T
    Ln = int(np.prod(L) / L[mode])
    C1 = C2 = 0
    for t in trange(1, T, desc=f'update B[{mode}]'):
        for j in range(Ln):
            C1 += _compute_b(F, cov_ZZt[t-1], j)
            C1 += EZ[t-1] @ np.outer(F[:, j], F[:, j]) @ EZ[t-1].T
            C2 += _compute_a(F, cov_ZZ_[t-1], j)
            C2 += np.outer(EZ[t, :, j], F[:, j]) @ EZ[t-1].T
    return C2 @ pinv(C1)

def _compute_a(F, cov, j):
    I, _, J, _ = cov.shape
    a = np.zeros((I, J))
    I, J = range(I), range(J)
    for p, q in itertools.product(I, J):
        a[p, q] = sum([F[k, j] * cov[p, j, q, k] for k in range(len(F))])
    return a

def _compute_b(F, cov, j):
    I, _, J, _ = cov.shape
    b = np.zeros((I, J))
    I, J = range(I), range(J)
    for p, q in itertools.product(I, J):
        for i, k in itertools.permutations(range(len(F)), 2):
            b[p, q] += F[k, j] * F[i, j] * cov[p, i, q, k]
    return b

def reshape_expectation(Ez, rank, mode):
    shape = (len(Ez), rank[mode], int(np.prod(rank)/rank[mode]))
    EZ = np.zeros(shape)
    for t, zt in enumerate(Ez):
        Zt = vec_to_tensor(zt, rank)
        Zt = np.moveaxis(Zt, mode, 0)
        EZ[t] = Zt.reshape(shape[1:])
    return EZ

def reshape_covariance(cov, rank, mode):
    M = len(rank)
    Lm = rank[mode]
    L_ = int(np.prod(rank) / Lm)
    covZZ = np.zeros((len(cov), Lm, L_, Lm, L_))
    for t, cov_t in enumerate(cov):
        cov_t = vec_to_tensor(cov_t, (*rank, *rank))
        cov_t = np.moveaxis(cov_t, mode, 0)
        cov_t = np.moveaxis(cov_t, mode + M, M)
        shape = (cov_t.shape[0], np.sum(cov_t.shape[1:M]),
                 cov_t.shape[M], np.sum(cov_t.shape[M + 1:2 * M]))
        covZZ[t] = cov_t.reshape(shape)
    return covZZ

def reconstruct_matrix(U, Z, mode):
    # Lemma 3.2
    ind = np.ones(len(U), dtype=bool)
    ind[mode] = False
    return np.dot(np.dot(U[mode], unfold(Z, mode)), kronecker(U[ind]).T)


if __name__ == '__main__':

    # load dataset
    X, _ = import_tensor('./dat/apple/')
    X = normalize_tensor(X)
    X = np.moveaxis(X, 1, -1)  # N_1 * ... * N_M * T

    # settings
    rank = [10, 5]
    weights = [.2, .2]

    # infer
    facets = Facets(X[:50, :, -100:], rank, weights)
    facets.em(max_iter=5)
    facets.save_params()
