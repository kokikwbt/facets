import numpy as np
import matplotlib.pyplot as plt
from tensorly.base import unfold
from tensorly.tenalg import kronecker
from tensorly.tucker_tensor import tucker_to_tensor

outdir = "./out/tmp/"
X = np.load(outdir+"X.npy")
T = X.shape[-1]
M = X.ndim - 1
z = np.loadtxt(outdir+"vec_z.txt")
U = [np.loadtxt(outdir+f"U_{i}.txt") for i in range(M)]
ranks = [U[i].shape[1] for i in range(M)]
Z = np.zeros((T, *ranks))
for t in range(T):
    Z[t] = z[t].reshape(ranks)

plt.plot(z)
plt.show()
for m in range(M):
    pred = np.zeros((T, X.shape[m]))
    for t in range(T):
        print(unfold(Z[t], m).shape, U[m].shape)
        print(kronecker(U, skip_matrix=m, reverse=True).T.shape)
        X_n = U[m] @ unfold(Z[t], m) @ kronecker(U, skip_matrix=m, reverse=True).T
        pred[t] = X_n[:, m]
    # plt.plot((X, -1)[:, i])
    plt.plot(pred)
    plt.show()

matU = kronecker(U, reverse=True)
predict = z @ matU.T
print(predict.shape)
print(unfold(X, -1).shape)
for i in range(unfold(X,-1).shape[1]):
    plt.plot(unfold(X, -1)[:, i])
    plt.plot(predict[:, i])
    plt.show()
exit()

# print(Z.shape)
# Z = np.moveaxis(Z, 1,2)
# print(Z.shape)
Xn = []
for t in range(T):
    Xn.append(tucker_to_tensor(Z[t], U))
Xn = np.array(Xn)
Xn = np.moveaxis(Xn, 0, -1)

# mode = 0
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         plt.plot(X[i, j, mode, :].T)
#         plt.plot(Xn[i, j, mode, :].T)
#         plt.show()
# exit()
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        plt.plot(X[i, j, :].T)
        plt.plot(Xn[i, j, :].T)
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
