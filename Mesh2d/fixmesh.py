import numpy as np


def triarea(p, t):
    """Helper function to calculate triangle areas"""
    p1 = p[t[:, 0], :]
    p2 = p[t[:, 1], :]
    p3 = p[t[:, 2], :]
    return 0.5 * ((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - 
                  (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1]))


def fixmesh(p, t, pfun=None, tfun=None):
    """
    Ensure that triangular mesh data is consistent.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        Array of nodal XY coordinates [x1,y1; x2,y2; etc]
    t : ndarray, shape (M, 3)
        Array of triangles as indices [n11,n12,n13; n21,n22,n23; etc]
    pfun : ndarray, shape (N, K), optional
        Nodal function values. Each column corresponds to a dependent function
    tfun : ndarray, shape (M, K), optional
        Triangle function values. Each column corresponds to a dependent function
        
    Returns:
    --------
    p : ndarray, shape (N', 2)
        Cleaned nodal coordinates
    t : ndarray, shape (M', 3)
        Cleaned triangle indices
    pfun : ndarray, shape (N', K) or None
        Cleaned nodal function values
    tfun : ndarray, shape (M', K) or None
        Cleaned triangle function values
    """
    
    TOL = 1.0e-10
    
    if p is None or t is None:
        raise ValueError('Wrong number of inputs')
    
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=int)
    
    if p.size != 2 * p.shape[0]:
        raise ValueError('P must be an Nx2 array')
    if t.size != 3 * t.shape[0]:
        raise ValueError('T must be an Mx3 array')
    if np.any(t < 0) or np.max(t) >= p.shape[0]:
        raise ValueError('Invalid T')
    
    if pfun is not None:
        pfun = np.asarray(pfun, dtype=float)
        if pfun.shape[0] != p.shape[0] or pfun.ndim != 2:
            raise ValueError('PFUN must be an NxK array')
    
    if tfun is not None:
        tfun = np.asarray(tfun, dtype=float)
        if tfun.shape[0] != t.shape[0] or tfun.ndim != 2:
            raise ValueError('TFUN must be an MxK array')
    
    # Remove duplicate nodes
    unique_nodes, unique_idx, inverse_idx = np.unique(p, axis=0, return_index=True, return_inverse=True)
    
    if pfun is not None:
        pfun = pfun[unique_idx, :]
    p = unique_nodes
    t = inverse_idx[t]
    
    # Triangle area
    A = triarea(p, t)
    Ai = A < 0.0
    Aj = np.abs(A) > TOL * np.linalg.norm(A, ord=np.inf)
    
    # Flip node numbering to give counter-clockwise order
    t[Ai, [0, 1]] = t[Ai, [1, 0]]
    
    # Remove zero area triangles
    t = t[Aj, :]
    if tfun is not None:
        tfun = tfun[Aj, :]
    
    # Remove un-used nodes
    used_nodes, node_map = np.unique(t.flatten(), return_inverse=True)
    
    if pfun is not None:
        pfun = pfun[used_nodes, :]
    p = p[used_nodes, :]
    t = node_map.reshape(t.shape)
    
    return p, t, pfun, tfun