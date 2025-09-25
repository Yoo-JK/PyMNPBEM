import numpy as np

def lgwt(N, a, b):
    """
    lgwt.py
    
    This script is for computing definite integrals using Legendre-Gauss 
    Quadrature. Computes the Legendre-Gauss nodes and weights on an interval
    [a,b] with truncation order N
    
    Suppose you have a continuous function f(x) which is defined on [a,b]
    which you can evaluate at any x in [a,b]. Simply evaluate it at all of
    the values contained in the x vector to obtain a vector f. Then compute
    the definite integral using sum(f*w);
    
    Written by Greg von Winckel - 02/25/2004
    Converted to Python
    
    Parameters:
    -----------
    N : int
        Truncation order (number of integration points)
    a : float
        Lower bound of integration interval
    b : float
        Upper bound of integration interval
        
    Returns:
    --------
    tuple
        x : np.ndarray
            Integration nodes on interval [a, b]
        w : np.ndarray
            Integration weights
    """
    N = N - 1
    N1 = N + 1
    N2 = N + 2
    
    xu = np.linspace(-1, 1, N1)
    
    # Initial guess
    y = (np.cos((2 * np.arange(N1) + 1) * np.pi / (2 * N + 2)) + 
         (0.27 / N1) * np.sin(np.pi * xu * N / N2))
    
    # Legendre-Gauss Vandermonde Matrix
    L = np.zeros((N1, N2))
    
    # Derivative of LGVM
    Lp = np.zeros((N1, N2))
    
    # Compute the zeros of the N+1 Legendre Polynomial
    # using the recursion relation and the Newton-Raphson method
    y0 = np.full_like(y, 2.0)
    
    # Iterate until new points are uniformly within epsilon of old points
    while np.max(np.abs(y - y0)) > np.finfo(float).eps:
        L[:, 0] = 1
        Lp[:, 0] = 0
        
        L[:, 1] = y
        Lp[:, 1] = 1
        
        for k in range(2, N2):
            L[:, k] = ((2 * k - 1) * y * L[:, k - 1] - (k - 1) * L[:, k - 2]) / k
        
        Lp = (N2) * (L[:, N1 - 1] - y * L[:, N2 - 1]) / (1 - y**2)
        
        y0 = y.copy()
        y = y0 - L[:, N2 - 1] / Lp
    
    # Linear map from [-1,1] to [a,b]
    x = (a * (1 - y) + b * (1 + y)) / 2
    
    # Compute the weights
    w = (b - a) / ((1 - y**2) * Lp**2) * (N2 / N1)**2
    
    return x, w