# Facets
Pyhton inplementation of:
1. Facets: Fast Comprehensive Mining of Coevolving High-order Time Series
\[[PDF](http://ycai.ws.gc.cuny.edu/files/2015/07/Facets_KDD15.pdf)\],
2. Fast Mining of a Network of Coevolving Time Series
\[[PDF](http://ycai.ws.gc.cuny.edu/files/2015/03/NoT_sdm15.pdf)\].

## Facets
Given a Network of High-order Time Series (Net-Hits),
this algorithm can recover its missing parts indicated by
the indicator tensor W or predict t time step after X.

```python
class facets.Facets
```
#### Methods

```python
__init__(self, X, ranks, weights)
```
##### Parameters:
- X: nd-array  
    - tensor of shape N_1 x N_2 x ... x T
- ranks: int list  
    - size of latent tensor Z
        (i.e., len(ranks) == tensor.ndim)
- weights: float list  
    - weight of contextual information for each mode of X.  
        if weight = 0, then the contextual information is ignored.  
        if weight = 1, then only the contextual information included to learn observation tensor U.


## DCMF
#### Usage
