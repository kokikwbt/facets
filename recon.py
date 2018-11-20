import numpy as np
import matplotlib.pyplot as plt
from tensorly.base import unfold
from tensorly.tenalg import kronecker

outdir = "./out/tmp/"
X = np.load(outdir+"X.npy")
z = np.loadtxt(outdir+"vec_z.txt")
U0 = np.loadtxt(outdir+"U_0.txt")
U1 = np.loadtxt(outdir+"U_1.txt")

rank = [4, 3]
T = X.shape[-1]
Z = z.reshape((T, *rank))
Z = np.moveaxis(Z, 1,2)
print(Z.shape)

print(U0.shape, U1.shape)
print(rank)

Xn = np.zeros((T, 9, 10))
for t in range(T):
    Xn[t] = U0 @ unfold(Z[t], 0) @ U1.T

# Xn = np.moveaxis(Xn, 1, 2)
print(X.shape)
print(Xn.shape)

for i in range(Xn.shape[2]):
    plt.figure()
    print(Xn.shape)
    plt.subplot(211)
    plt.plot(X[:, i, :].T)
    plt.subplot(212)
    plt.plot(Xn[:, :, i])
plt.show()
