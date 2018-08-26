import numpy as np
import pandas as pd
import pycountry

countries = list(pycountry.countries)
excepts = [
    'AI','AS','AG','BI','BT','CF','CK','KY','ER','FM',
    'GW','IO','MH','MS','NF','NR','SH'
]

def import_trends(indir, geo_level='country'):
    _geo = 'US' if geo_level == 'country' else ' '
    _fn = f'{indir}{_geo}.csv'
    _df = pd.read_csv(_fn)
    _mat = _df.as_matrix(_df.keys()[1:])
    X = []  # search volumes: t * m
    L = []  # locations: l * 1
    for geo in countries:
        if geo.alpha_2 in excepts: continue
        fn = f'{indir}{geo.alpha_2}.csv'
        df = pd.read_csv(fn)
        mat = df.as_matrix(df.keys()[1:])
        if not mat.shape == _mat.shape: continue
        mat[mat==0] = np.nan
        X.append(np.expand_dims(mat, axis=0))
        L.append(geo)
    X = np.concatenate(X, axis=0)
    return X, L

def import_tensor(indir):
    # X (geo, time, key)
    X = import_trends(indir)
    return X

def normalize_tensor(X):
    g, t, k = X.shape
    for i in range(g):
        # X[i] = normalize(X[i], axis=1)
        for j in range(k):
            tmp = X[i][:, j]
            idx = tmp > 0
            if not len(tmp[idx]):
                continue
            mx = tmp[idx].max()
            mn = tmp[idx].min()
            X[i][idx, j] = (X[i][idx, j] - mn) / (mx - mn)
    return X
