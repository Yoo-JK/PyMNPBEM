import numpy as np

def pdist2(p1, p2):
    """
    PDIST2 - Distance array between positions P1 and P2.
    
    Parameters:
    -----------
    p1 : array_like
        First position array (M x D)
    p2 : array_like  
        Second position array (N x D)
        
    Returns:
    --------
    np.ndarray
        Distance array between P1 and P2 (M x N)
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    
    # Ensure 2D arrays
    if p1.ndim == 1:
        p1 = p1.reshape(1, -1)
    if p2.ndim == 1:
        p2 = p2.reshape(1, -1)
    
    # Square of distance using broadcasting
    p1_sq = np.sum(p1**2, axis=1, keepdims=True)  # (M, 1)
    p2_sq = np.sum(p2**2, axis=1)                 # (N,)
    
    d = p1_sq + p2_sq - 2 * p1 @ p2.T
    
    # Avoid rounding errors
    d[d < 1e-10] = 0
    
    # Square root of distance
    d = np.sqrt(d)
    
    return d