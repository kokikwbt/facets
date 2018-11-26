import numpy as np
import tensorly as tl
from myplot import heatmap

ranks = [2, 3, 5]
M = len(ranks)
X = np.arange(np.prod(ranks))
X = np.reshape(X, ranks)

cov = np.outer(X, X)
# cov = np.reshape(cov, (*ranks, *ranks))
# for i in range(ranks[1]):
#     print(cov[:, i, 0, :, i, 0])
    # heatmap(cov[:, i, 1, :, i, ], show=True)
# cov = np.reshape(cov, (2,2,3,3,5,5))
cov = np.reshape(cov, (2,3,5,5,3,2))

print(cov.shape)

# simulate mode-1 matricize
mode = 1
Nm = ranks[mode]
Nn = int(np.prod(ranks) / Nm)
cov = np.moveaxis(cov, mode, 0)
cov = np.moveaxis(cov, mode + M, 0 + M)
print(cov.shape)
cov = np.reshape(cov, (Nm, Nn, Nm, Nn))
print(cov.shape)
print(cov)
sum = np.sum(cov, axis=1)
sum = np.sum(sum, axis=2)
print(sum, sum.shape)
heatmap(sum, show=True)

# print(tl.tensor_to_vec(X))
# print(X, X.shape)
