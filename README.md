# Facets

This is a Python implementation for
1. Facets: Fast Comprehensive Mining of Coevolving High-order Time Series
\[[download](http://ycai.ws.gc.cuny.edu/files/2015/07/Facets_KDD15.pdf)\],
2. Fast Mining of a Network of Coevolving Time Series
\[[download](http://ycai.ws.gc.cuny.edu/files/2015/03/NoT_sdm15.pdf)\].

## Facets
Given a N-dimensional tensor series,
`facets.Facets(X, ranks, weights)`
Parameters:
    - X: nd-array
        tensor of shape N_1 x N_2 x ... x T
    - ranks: int list
        size of latent tensor Z
        (len(ranks) == tensor.ndim)
    - weights: float list
        weight of contextual information for each mode of X.
        if weight = 0, then the contextual information is ignored. if weight = 1, then only the contextual information included to learn observation tensor U.

###### example
```python
  # X: nd-array ()
  # rank:
  # weights:
  facets = Facets(X, rank, weights)
  facets.em(max_iter=20)
  facets.save_params()
```

`$ python3 facets.py`

## DCMF
#### Usage
`$ python3 dcmf.py`
