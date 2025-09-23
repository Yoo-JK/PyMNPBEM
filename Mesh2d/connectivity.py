import numpy as np


def connectivity(p, t):
    """
    Assemble connectivity data for a triangular mesh.
    
    The edge based connectivity is built for a triangular mesh and the
    boundary nodes identified. This data should be useful when implementing
    FE/FV methods using triangular meshes.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        Array of node coordinates [x1,y1; x2,y2; etc]
    t : ndarray, shape (M, 3)
        Array of triangles as indices [n11,n12,n13; n21,n22,n23; etc]
        
    Returns:
    --------
    e : ndarray, shape (K, 2)
        Array of unique mesh edges [n11,n12; n21,n22; etc]
    te : ndarray, shape (M, 3)
        Array of triangles as indices into E [e11,e12,e13; e21,e22,e23; etc]
    e2t : ndarray, shape (K, 2)
        Array of triangle neighbours for unique mesh edges [t11,t12; t21,t22; etc]
        Each row has two entries corresponding to the triangle numbers
        associated with each edge in E. Boundary edges have e2t[i,1] = 0.
    bnd : ndarray, shape (N,), dtype=bool
        Logical array identifying boundary nodes. p[i,:] is a boundary
        node if bnd[i] = True.
    """
    
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
    
    # Unique mesh edges as indices into P
    numt = t.shape[0]
    vect = np.arange(numt)  # Triangle indices (0-based)
    
    # Create all edges - not unique yet
    e = np.vstack([
        t[:, [0, 1]],  # Edge 0-1
        t[:, [1, 2]],  # Edge 1-2
        t[:, [2, 0]]   # Edge 2-0
    ])
    
    # Get unique edges
    sorted_edges = np.sort(e, axis=1)
    unique_edges, unique_indices, inverse_indices = np.unique(
        sorted_edges, axis=0, return_index=True, return_inverse=True
    )
    
    e = unique_edges
    
    # Unique edges in each triangle
    te = np.column_stack([
        inverse_indices[vect],
        inverse_indices[vect + numt],
        inverse_indices[vect + 2 * numt]
    ])
    
    # Edge-to-triangle connectivity
    # Each row has two entries corresponding to the triangle numbers
    # associated with each edge. Boundary edges have e2t[i,1] = 0.
    nume = e.shape[0]
    e2t = np.zeros((nume, 2), dtype=int)
    
    for k in range(numt):
        for j in range(3):
            ce = te[k, j]
            if e2t[ce, 0] == 0:
                e2t[ce, 0] = k + 1  # Convert to 1-based for MATLAB compatibility
            else:
                e2t[ce, 1] = k + 1  # Convert to 1-based for MATLAB compatibility
    
    # Flag boundary nodes
    bnd = np.zeros(p.shape[0], dtype=bool)
    boundary_edges = e[e2t[:, 1] == 0]  # Edges with only one triangle
    bnd[boundary_edges.flatten()] = True
    
    return e, te, e2t, bnd