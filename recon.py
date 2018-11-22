import numpy as np
import matplotlib.pyplot as plt
from tensorly.base import unfold
from tensorly.tenalg import kronecker
from tensorly.tucker_tensor import tucker_to_tensor

outdir = "./out/tmp/"
X = np.load(outdir+"X.npy")
z = np.loadtxt(outdir+"vec_z.txt")
U0 = np.loadtxt(outdir+"U_0.txt")
U1 = np.loadtxt(outdir+"U_1.txt")

rank = [3, 4]
T = X.shape[-1]
Z = np.zeros((T, 4,3))
for t in range(T):
    Z[t] = z[t].reshape((4,3))
# print(Z.shape)
# Z = np.moveaxis(Z, 1,2)
# print(Z.shape)
Xn = []
for t in range(T):
    Xn.append(tucker_to_tensor(Z[t], [U0, U1]))
Xn = np.array(Xn)
print(U0.shape, U1.shape)
print(rank)

# Xn = np.zeros((T, 9, 10))
# for t in range(T):
#     Xn[t] = U0 @ unfold(Z[t], 0) @ U1.T

for i in range(X.shape[0]):
    plt.subplot(211)
    plt.plot(X[i, :, :].T)
    plt.subplot(212)
    plt.plot(Xn[:, i, :])
    plt.show()

exit()
for j in range(Xn.shape[1]):
    for i in range(Xn.shape[2]):
        plt.figure()
        plt.subplot(211)
        plt.plot(X[j, i, :])
        plt.plot(Xn[:, j, i])
        # plt.ylim([-0.2, 1])
    plt.show()
