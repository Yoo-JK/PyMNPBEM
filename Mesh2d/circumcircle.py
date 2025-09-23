import numpy as np


def circumcircle(p, t):
    """
    XY centre co-ordinates and radius of triangle circumcircles.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        Array of nodal XY co-ordinates
    t : ndarray, shape (M, 3)
        Array of triangles as indices into P
        
    Returns:
    --------
    cc : ndarray, shape (M, 3)
        Array of circumcircles where:
        cc[:, 0:2] = XY center coordinates
        cc[:, 2] = R^2 (radius squared)
    """
    
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=int)
    
    # Initialize output with same shape as t
    cc = np.zeros(t.shape, dtype=float)
    
    # Corner XY coordinates
    p1 = p[t[:, 0], :]
    p2 = p[t[:, 1], :]
    p3 = p[t[:, 2], :]
    
    # Set equation for center of each circumcircle:
    # [a11, a12; a21, a22] * [x; y] = [b1; b2] * 0.5
    a1 = p2 - p1
    a2 = p3 - p1
    
    b1 = np.sum(a1 * (p2 + p1), axis=1)
    b2 = np.sum(a2 * (p3 + p1), axis=1)
    
    # Explicit inversion
    idet = 0.5 / (a1[:, 0] * a2[:, 1] - a2[:, 0] * a1[:, 1])
    
    # Circumcentre XY
    cc[:, 0] = (a2[:, 1] * b1 - a1[:, 1] * b2) * idet
    cc[:, 1] = (-a2[:, 0] * b1 + a1[:, 0] * b2) * idet
    
    # Radius^2
    cc[:, 2] = np.sum((p1 - cc[:, 0:2])**2, axis=1)
    
    return cc