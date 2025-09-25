import numpy as np

def lglnodes(n):
    """
    LGLNODES - Legendre-Gauss-Lobatto nodes for integration.
    Adapted from Greg von Winckel.
    
    Parameters:
    -----------
    n : int
        Number of integration points
        
    Returns:
    --------
    tuple
        x : np.ndarray
            Integration points in interval [-1, 1]
        w : np.ndarray
            Integration weights
    """
    # Truncation + 1
    n1 = n + 1
    
    # Use the Chebyshev-Gauss-Lobatto nodes as the first guess
    x = np.cos(np.pi * np.arange(n1) / n)
    
    # Legendre Vandermonde Matrix
    p = np.zeros((n1, n1))
    
    # Compute P_(N) using the recursion relation.
    # Compute its first and second derivatives and 
    # update x using the Newton-Raphson method.
    xold = np.full_like(x, 2.0)
    
    while np.max(np.abs(x - xold)) > np.finfo(float).eps:
        xold = x.copy()
        p[:, 0] = 1.0    
        p[:, 1] = x
        
        for k in range(2, n + 1):
            p[:, k] = ((2 * k - 1) * x * p[:, k - 1] - (k - 1) * p[:, k - 2]) / k
        
        x = xold - (x * p[:, n] - p[:, n - 1]) / (n1 * p[:, n])
    
    w = 2.0 / (n * n1 * p[:, n]**2)
    
    return x, w