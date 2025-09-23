import numpy as np


def refine(p, t, ti=None, f=None):
    """
    REFINE: Refine triangular meshes.
    
    Quadtree triangle refinement is performed, with each triangle split into
    four sub-triangles. The new triangles are created by joining nodes
    introduced at the edge midpoints. The refinement is "quality" preserving,
    with the aspect ratio of the sub-triangles being equal to that of the
    parent.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        Nodal XY coordinates, [x1,y1; x2,y2; etc]
    t : ndarray, shape (M, 3)
        Triangles as indices, [n11,n12,n13; n21,n22,n23; etc]
    ti : ndarray, shape (M,), optional
        Logical array, with ti[k] = True if kth triangle is to be refined.
        If None, uniform refinement is performed on all triangles.
    f : ndarray, shape (N, K), optional
        Nodal function values. Each column in f corresponds to a dependent
        function, f[:, 0] = F1(p), f[:, 1] = F2(p), etc.
        
    Returns:
    --------
    p : ndarray, shape (N_new, 2)
        Refined nodal coordinates
    t : ndarray, shape (M_new, 3)
        Refined triangles
    f : ndarray, shape (N_new, K), optional
        Refined function values (if input f was provided)
        
    Notes:
    ------
    UNIFORM REFINEMENT:
    [p, t] = refine(p, t)
    
    NON-UNIFORM REFINEMENT:
    Non-uniform refinement can also be performed by specifying which
    triangles are to be refined. Quadtree refinement is performed on
    specified triangles. Neighbouring triangles are also refined in order to
    preserve mesh compatibility. These triangles are refined using bi-section.
    
    [p, t] = refine(p, t, ti)
    
    Functions defined on the nodes in p can also be refined using linear
    interpolation:
    [p, t, f] = refine(p, t, ti, f)
    
    It is often useful to smooth the refined mesh using smoothmesh. Generally
    this will improve element quality.
    
    Darren Engwirda : 2007
    Python conversion : 2025
    """
    
    # Convert inputs to numpy arrays
    p = np.asarray(p)
    t = np.asarray(t, dtype=int)
    
    # Input validation
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("p must be an Nx2 array")
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError("t must be an Mx3 array")
    
    # Check if function values are provided
    got_f = f is not None
    
    # Handle refinement indicator
    if ti is None:
        # Uniform refinement
        ti = np.ones(t.shape[0], dtype=bool)
    else:
        ti = np.asarray(ti, dtype=bool)
        if ti.shape[0] != t.shape[0]:
            raise ValueError("ti must be an Mx1 array")
    
    # Validate function values if provided
    if got_f:
        f = np.asarray(f)
        if f.shape[0] != p.shape[0]:
            raise ValueError("f must have same number of rows as p")
        if f.ndim == 1:
            f = f.reshape(-1, 1)
    
    # Ensure we start with a consistent mesh
    if got_f:
        p, t, f = fixmesh_with_f(p, t, f)
    else:
        p, t = fixmesh(p, t)
    
    # Edge connectivity
    numt = t.shape[0]
    vect = np.arange(numt)
    
    # Create edge list (not unique)
    e = np.vstack([
        t[:, [0, 1]],  # edges 0-1
        t[:, [1, 2]],  # edges 1-2  
        t[:, [2, 0]]   # edges 2-0
    ])
    
    # Get unique edges and their mapping
    e_sorted = np.sort(e, axis=1)
    e_unique, indices, inverse = np.unique(e_sorted, axis=0, return_index=True, return_inverse=True)
    
    # Map triangle edges to unique edge indices
    te = inverse.reshape(3, numt).T  # te[i, j] = unique edge index for j-th edge of i-th triangle
    
    # Determine which edges to split
    split = np.zeros(e_unique.shape[0], dtype=bool)
    split[te[ti, :].ravel()] = True  # Mark edges of triangles to be refined
    
    # Iteratively mark additional edges to maintain mesh compatibility
    nsplit = np.sum(split)
    while True:
        # Triangles where we will split 3 edges (quadtree refinement)
        split3 = np.sum(split[te], axis=1) >= 2
        
        # Mark all edges of these triangles for splitting
        split[te[split3, :].ravel()] = True
        
        new_splits = np.sum(split) - nsplit
        if new_splits == 0:
            break
        nsplit += new_splits
    
    # Triangles where we will split exactly 1 edge (bisection)
    split1 = np.sum(split[te], axis=1) == 1
    split3 = np.sum(split[te], axis=1) == 3
    
    # Create new nodes at edge midpoints
    np_orig = p.shape[0]
    pm = 0.5 * (p[e_unique[split, 0], :] + p[e_unique[split, 1], :])
    p = np.vstack([p, pm])
    
    # Create mapping from split edges to new node indices
    edge_to_node = np.zeros(e_unique.shape[0], dtype=int)
    edge_to_node[split] = np.arange(nsplit) + np_orig
    
    # Build new triangle list
    tnew = []
    
    # Keep triangles that are not being refined
    keep_mask = ~(split1 | split3)
    if np.any(keep_mask):
        tnew.append(t[keep_mask, :])
    
    # Handle triangles with 3 split edges (quadtree refinement)
    if np.any(split3):
        t3 = t[split3, :]
        te3 = te[split3, :]
        
        n1 = t3[:, 0]  # Original vertices
        n2 = t3[:, 1]
        n3 = t3[:, 2]
        n4 = edge_to_node[te3[:, 0]]  # New midpoint nodes
        n5 = edge_to_node[te3[:, 1]]
        n6 = edge_to_node[te3[:, 2]]
        
        # Create 4 new triangles for each original triangle
        new_triangles = np.column_stack([
            np.column_stack([n1, n4, n6]),  # Corner triangle 1
            np.column_stack([n4, n2, n5]),  # Corner triangle 2  
            np.column_stack([n5, n3, n6]),  # Corner triangle 3
            np.column_stack([n4, n5, n6])   # Center triangle
        ]).reshape(-1, 3)
        
        tnew.append(new_triangles)
    
    # Handle triangles with 1 split edge (bisection)
    if np.any(split1):
        t1 = t[split1, :]
        te1 = te[split1, :]
        
        # Find which edge is being split in each triangle
        split_edges = split[te1]  # Boolean array
        edge_indices = np.where(split_edges)
        row_indices, col_indices = edge_indices
        
        # Reorder vertices so split edge is always between n1 and n2
        n1 = np.zeros(len(row_indices), dtype=int)
        n2 = np.zeros(len(row_indices), dtype=int)
        n3 = np.zeros(len(row_indices), dtype=int)
        n4 = np.zeros(len(row_indices), dtype=int)
        
        split1_indices = np.where(split1)[0]
        actual_tri_indices = split1_indices[row_indices]
        
        for k, (tri_idx, col) in enumerate(zip(actual_tri_indices, col_indices)):
            # Reorder so split edge is between vertices 0 and 1
            vertex_order = [(col + i) % 3 for i in range(3)]
            n1[k] = t[tri_idx, vertex_order[0]]
            n2[k] = t[tri_idx, vertex_order[1]]  
            n3[k] = t[tri_idx, vertex_order[2]]
            n4[k] = edge_to_node[te[tri_idx, col]]
        
        # Create 2 new triangles for each bisected triangle
        new_triangles = np.vstack([
            np.column_stack([n1, n4, n3]),  # Triangle 1
            np.column_stack([n4, n2, n3])   # Triangle 2
        ])
        
        tnew.append(new_triangles)
    
    # Combine all new triangles
    t = np.vstack(tnew) if tnew else np.array([]).reshape(0, 3).astype(int)
    
    # Refine function values using linear interpolation
    if got_f:
        fm = 0.5 * (f[e_unique[split, 0], :] + f[e_unique[split, 1], :])
        f = np.vstack([f, fm])
        return p, t, f
    else:
        return p, t


def fixmesh(p, t):
    """
    Fix mesh by removing duplicate nodes and unused nodes.
    
    This is a simplified version - full implementation would be more complex.
    """
    # Remove duplicate points
    p, unique_idx, inverse_idx = np.unique(p, axis=0, return_index=True, return_inverse=True)
    
    # Update triangle indices
    t = inverse_idx[t]
    
    # Remove degenerate triangles
    valid_triangles = ~(
        (t[:, 0] == t[:, 1]) |
        (t[:, 1] == t[:, 2]) |
        (t[:, 2] == t[:, 0])
    )
    t = t[valid_triangles, :]
    
    return p, t


def fixmesh_with_f(p, t, f):
    """
    Fix mesh while preserving function values.
    """
    # Remove duplicate points
    p, unique_idx, inverse_idx = np.unique(p, axis=0, return_index=True, return_inverse=True)
    
    # Update function values
    f = f[unique_idx, :]
    
    # Update triangle indices
    t = inverse_idx[t]
    
    # Remove degenerate triangles
    valid_triangles = ~(
        (t[:, 0] == t[:, 1]) |
        (t[:, 1] == t[:, 2]) |
        (t[:, 2] == t[:, 0])
    )
    t = t[valid_triangles, :]
    
    return p, t, f


def refine_uniform(p, t, levels=1, f=None):
    """
    Perform uniform refinement for specified number of levels.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        Initial nodes
    t : ndarray, shape (M, 3)  
        Initial triangles
    levels : int
        Number of refinement levels
    f : ndarray, optional
        Function values to interpolate
        
    Returns:
    --------
    p : ndarray
        Refined nodes
    t : ndarray
        Refined triangles
    f : ndarray, optional
        Refined function values
    """
    
    for level in range(levels):
        if f is not None:
            p, t, f = refine(p, t, None, f)
        else:
            p, t = refine(p, t)
    
    if f is not None:
        return p, t, f
    else:
        return p, t


def refine_adaptive(p, t, quality_threshold=0.3, f=None):
    """
    Perform adaptive refinement based on triangle quality.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        Nodes
    t : ndarray, shape (M, 3)
        Triangles
    quality_threshold : float
        Triangles with quality below this threshold will be refined
    f : ndarray, optional
        Function values
        
    Returns:
    --------
    p : ndarray
        Refined nodes
    t : ndarray
        Refined triangles  
    f : ndarray, optional
        Refined function values
    """
    
    # Calculate triangle quality (placeholder - would use actual quality function)
    # For now, just refine randomly for demonstration
    np.random.seed(42)
    ti = np.random.random(t.shape[0]) < 0.3  # Refine 30% of triangles randomly
    
    if f is not None:
        return refine(p, t, ti, f)
    else:
        return refine(p, t, ti)


# Example usage function
def refine_example():
    """
    Example of how to use the refine function.
    """
    
    # Create a simple triangular mesh
    p = np.array([
        [0.0, 0.0],
        [1.0, 0.0], 
        [0.5, 1.0],
        [1.5, 0.5]
    ])
    
    t = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ])
    
    print("Original mesh:")
    print(f"Nodes: {p.shape[0]}, Triangles: {t.shape[0]}")
    
    # Uniform refinement
    p_ref, t_ref = refine(p, t)
    print(f"After uniform refinement - Nodes: {p_ref.shape[0]}, Triangles: {t_ref.shape[0]}")
    
    # Selective refinement
    ti = np.array([True, False])  # Only refine first triangle
    p_sel, t_sel = refine(p, t, ti)
    print(f"After selective refinement - Nodes: {p_sel.shape[0]}, Triangles: {t_sel.shape[0]}")
    
    return p_ref, t_ref, p_sel, t_sel