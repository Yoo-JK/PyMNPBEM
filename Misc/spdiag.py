import numpy as np
from scipy.sparse import diags

def spdiag(a):
    """
    SPDIAG - Put array values on the diagonal of a sparse matrix.
    
    Parameters:
    -----------
    a : array_like
        Array values to place on diagonal
        
    Returns:
    --------
    scipy.sparse matrix
        Sparse diagonal matrix with values from a
    """
    a = np.asarray(a).flatten()
    n = len(a)
    return diags(a, offsets=0, shape=(n, n), format='csr')