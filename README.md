# Facets

This is a Python implementation for
1. Facets: Fast Comprehensive Mining of Coevolving High-order Time Series
\[[download](http://ycai.ws.gc.cuny.edu/files/2015/07/Facets_KDD15.pdf)\],
2. Fast Mining of a Network of Coevolving Time Series
\[[download](http://ycai.ws.gc.cuny.edu/files/2015/03/NoT_sdm15.pdf)\].

## facets.py

#### Quick usage
```python
  # X: nd-array ()
  # rank:
  # weights:
  facets = Facets(X, rank, weights)
  facets.em(max_iter=20)
  facets.save_params()
```

`$ python3 facets.py`

## dcmf.py
#### Usage
`$ python3 dcmf.py`
