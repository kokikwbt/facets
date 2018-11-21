import os
import shutil
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
from scipy.linalg import pinv
from tqdm import tqdm, trange
from myplot import *
warnings.filterwarnings("ignore")

class DCMF(object):
    """
    Dynamic Contextual Matrix Factorization
    """
    def __init__(self, X, rank, weight=None):
        if not X.ndim == 2:
            raise ValueError("input must be 2D-array")
        # Given inputs:
        self.T = T = X.shape[0]
        self.n = n = X.shape[1]
        self.l = l = rank if rank < n else n
        self.X = X
        self.W = ~np.isnan(X)
        self.S = np.random.rand(n, n)
        self.lmd = 1. if not weight else weight
        # Model parameters:
        self.U = np.random.rand(n, l)
        self.B = np.random.rand(l, l)
        self.z0 = np.random.rand(l)
        self.psi0 = np.random.rand(l, l)
        self.sgmZ = np.random.rand()
        self.sgmX = np.random.rand()
        self.sgmS = np.random.rand()
        self.sgmV = np.random.rand()

    def em(self, max_iter=100, tol=1.e-7, logging=True):
        T, n, l = self.T, self.n, self.l
        if logging: self._init_logs()
        for iteration in range(max_iter):
            if logging: self._update_log()
            print("=========")
            print(" iter", iteration)
            print("=========")
            """
            E-step
            ======
            """
            mu_, psi, K, P = forward(
                self.l, self.X, self.W, self.U, self.B,
                self.z0, self.psi0, self.sgmX, self.sgmZ
            )
            zt, zt_, ztt = backward(
                self.l, self.X, self.W, self.U, self.B,
                self.z0, self.psi0, self.sgmX, self.sgmZ,
                mu_, psi, K, P
            )
            # Eq. (3.12)
            v = np.zeros((n, l))
            vv = np.zeros((n, l, l))
            M = self.U.T @ self.U + self.sgmS / self.sgmV * np.eye(self.l)
            Minv = pinv(M)
            gamma = self.sgmS * Minv
            for j in trange(n, desc='compute p(v|s)'):
                v[j] = Minv @ self.U.T @ self.S[j, :]
                vv[j] = gamma + np.outer(v[j], v[j])

            """
            M-step
            ======
            """
            self.z0 = zt[0]
            self.psi0 = ztt[0] - np.outer(zt[0], zt[0])
            self.B = B = sum(zt_[1:]) @ pinv(sum(ztt[:-1]))
            self.sgmV = sum([np.trace(vv[j]) for j in range(n)]) / (n * l)

            res = np.trace(
                sum(ztt[1:])
                - sum(zt_[1:]) @ B.T
                - (sum(zt_[1:]) @ B.T).T
                + B @ sum(ztt[:-1]) @ B.T
            )
            self.sgmZ = res / ((T - 1) * l)

            for i in range(n):
                A1 = self.lmd / self.sgmS * sum([self.S[i, j] * v[j] for j in range(n)])
                A1 += (1 - self.lmd) / self.sgmX * sum([self.W[t, i] * self.X[t, i] * zt[t] for t in range(T)])
                A2 = self.lmd / self.sgmS * sum(vv)
                A2 += (1 - self.lmd) / self.sgmX * sum([self.W[t, i] * ztt[t] for t in range(T)])
                self.U[i, :] = A1 @ pinv(A2)

            U, S = self.U, self.S
            res = sum([S[j].T @ S[j] - 2 * S[j] @ (U @ v[j]) for j in range(n)])
            res += np.trace(U @ sum(vv) @ U.T)
            self.sgmS = res / n ** 2

            res = 0
            for t in range(T):
                ot = self.W[t, :]
                xt = self.X[t, ot]
                Ht = self.U[ot, :]
                res += np.trace(Ht @ ztt[t] @ Ht.T)
                res += xt @ xt - 2 * xt @ (Ht @ zt[t])
            self.sgmX = res / self.W.sum()

            print('===> sgmV:', self.sgmV)
            print('===> sgmX:', self.sgmX)
            print('===> sgmZ:', self.sgmZ)
            print('===> sgmS:', self.sgmS)

        self.Z = zt
        self.V = np.vstack(v)
        self.recon_ = (self.U @ self.Z.T).T
        self.rmse = np.sqrt(mean_squared_error(self.X, self.recon_))

    def _compute_log_likelihood(self):
        llh = 0
        # temporal smoothness
        # observed time series
        # contextual network
        self.llh = llh
        return llh

    def _init_logs(self):
        self.U_log = []
        self.B_log = []
        self.z0_log = []
        self.psi0_log = []
        self.sgmV_log = []
        self.sgmX_log = []
        self.sgmZ_log = []
        self.sgmS_log = []

    def _update_log(self):
        self.U_log.append(self.U)
        self.B_log.append(self.B)
        self.z0_log.append(self.z0)
        self.psi0_log.append(self.psi0)
        self.sgmV_log.append(self.sgmV)
        self.sgmX_log.append(self.sgmX)
        self.sgmZ_log.append(self.sgmZ)
        self.sgmS_log.append(self.sgmS)

    def save_model(self, outdir='./out/'):
        _save_model(self, outdir)

def forward(l, X, W, U, B, z0, psi0, sgmX, sgmZ):
    """
    ot: the indices of the observed entries of xt
    Ht: the corresponding compressed version of U
    x*(t) = H(t) @ z(t) + Gaussian noise
    """
    T, n = X.shape
    mu_ = np.zeros((T, l))
    psi = np.zeros((T, l, l))
    K = np.zeros((T, l, n))
    P = np.zeros((T, l, l))
    for t in trange(T, desc='forward'):
        # construct H(t) based on Eq. (3.6)
        ot = W[t, :]
        lt = sum(ot)
        x = X[t, ot]
        H = U[ot, :]
        # Estimate mu and phi: Eq. (3.8), (3.9)
        if t == 0:
            K[0] = psi0 @ H.T @ pinv(H @ psi0 @ H.T + sgmX * np.eye(lt))
            psi[0] = (np.eye(l) - K[0] @ H) @ psi0
            mu_[0] = z0 + K[0] @ (x - H @ z0)
        else:
            P[t-1] = B @ psi[t-1] @ B.T + sgmZ * np.eye(l)
            K[t] = P[t-1] @ H.T @ pinv(H @ P[t-1] @ H.T + sgmX * np.eye(lt))
            psi[t] = (np.eye(l) - K[t] @ H) @ P[t-1]
            mu_[t] = B @ mu_[t-1] + K[t] @ (x - H @ B @ mu_[t-1])
    return mu_, psi, K, P

def backward(l, X, W, U, B, z0, psi0, sgmX, sgmZ, mu_, psi, K, P):
    T, n = X.shape
    J = np.zeros((T, l, l))
    zt = np.zeros((T, l))
    zt_ = np.zeros((T, l, l))
    ztt = np.zeros((T, l, l))
    mu_h = np.zeros((T, l))
    psih = np.zeros((T, l, l))
    mu_h[-1] = mu_[-1]
    psih[-1] = psi[-1]
    for t in tqdm(list(reversed(range(T - 1))), desc='backward'):
        J[t] = psi[t] @ B.T @ pinv(P[t])
        psih[t] = psi[t] + J[t] @ (psih[t+1] - P[t]) @ J[t].T
        mu_h[t] = mu_[t] + J[t] @ (mu_h[t+1] - B @ mu_[t])
    for t in trange(T, desc="compute E[z] & E[zz']"):
        if t > 0:
            zt_[t] = psih[t] @ J[t-1].T + np.outer(mu_h[t], mu_h[t-1])
        ztt[t] = psih[t] + np.outer(mu_h[t], mu_h[t])
    zt = mu_h
    return zt, zt_, ztt

def _save_model(dcmf, outdir, viz=True):
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)
    os.mkdir(outdir+"params/")
    np.savetxt(f"{outdir}params/U.txt", dcmf.U)
    np.savetxt(f"{outdir}params/B.txt", dcmf.B)
    np.savetxt(f"{outdir}params/z_0.txt", dcmf.z0)
    np.savetxt(f"{outdir}params/psi_0.txt", dcmf.psi0)
    with open(f"{outdir}params/covars.txt", "w") as f:
        f.write(f"sigma_v, {dcmf.sgmV}\n")
        f.write(f"sigma_s, {dcmf.sgmS}\n")
        f.write(f"sigma_x, {dcmf.sgmX}\n")
        f.write(f"sigma_z, {dcmf.sgmZ}\n")

    if viz:
        os.mkdir(outdir+"viz/")
        heatmap(dcmf.U, outfn=f'{outdir}/viz/U.png')
        heatmap(dcmf.B, outfn=f'{outdir}/viz/B.png')
        heatmap(dcmf.psi0, outfn=f'{outdir}/viz/psi_0.png')
        plot(dcmf.z0_log, title='z_0',
            xlabel='# of iter (EM)', ylabel='Value',
            outfn=f'{outdir}viz/z0.png')
        # plot(dcmf.psi0_log, title='psi_0',
        #      xlabel='# of iter (EM)', ylabel='Value',
        #      outfn=f'{outdir}psi0.png')
        plot(dcmf.sgmX_log, title='sigma_X',
            xlabel='# of iter (EM)', ylabel='Value',
            outfn=f'{outdir}viz/sigma_x.png')
        plot(dcmf.sgmV_log, title='sigma_V',
            xlabel='# of iter (EM)', ylabel='Value',
            outfn=f'{outdir}viz/sigma_v.png')
        plot(dcmf.sgmZ_log, title='sigma_Z',
            xlabel='# of iter (EM)', ylabel='Value',
            outfn=f'{outdir}viz/sigma_z.png')
        plot(dcmf.sgmS_log, title='sigma_S',
            xlabel='# of iter (EM)', ylabel='Value',
            outfn=f'{outdir}viz/sigma_s.png')
        fit_plot(dcmf.X, dcmf.recon_,
            outfn=f'{outdir}viz/fit.png')
        fit_scatter(dcmf.X, dcmf.recon_,
            outfn=f'{outdir}viz/fit')

if __name__ == '__main__':

    X = np.loadtxt('./dat/86_11.amc.4d', delimiter=',')
    X = scale(X)

    model = DCMF(X, 4, weight=.8)
    model.em(max_iter=20)
    print('rmse:', model.rmse)
    model.save_model()
