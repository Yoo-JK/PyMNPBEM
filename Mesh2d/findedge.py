import numpy as np


def findedge(p, node, edge=None, TOL=1.0e-12):
    """
    Locate points on edges.
    
    Determine which edges a series of points lie on in a 2D plane.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        Array of xy co-ordinates of points to be checked
    node : ndarray, shape (K, 2)
        Array of xy co-ordinates of edge endpoints
    edge : ndarray, shape (M, 2), optional
        Array of edges, specified as connections between vertices in NODE
    TOL : float, optional
        Tolerance used when testing points (default: 1.0e-12)
        
    Returns:
    --------
    enum : ndarray, shape (N,), dtype=int
        Array of edge numbers (1-based), corresponding to the edge that each
        point lies on. Points that do not lie on any edges are assigned 0.
    """
    
    if p is None or node is None:
        raise ValueError('Insufficient inputs')
    
    p = np.asarray(p, dtype=float)
    node = np.asarray(node, dtype=float)
    
    if p.shape[1] != 2:
        raise ValueError('P must be an Nx2 array.')
    if node.shape[1] != 2:
        raise ValueError('NODE must be an Mx2 array.')
    
    nnode = node.shape[0]
    
    # Build edge if not passed
    if edge is None:
        edge = np.column_stack([
            np.arange(1, nnode),     # 1 to nnode-1
            np.arange(2, nnode + 1)  # 2 to nnode
        ])
        edge = np.vstack([edge, [nnode, 1]])  # Add closing edge
        edge = edge - 1  # Convert to 0-based indexing
    else:
        edge = np.asarray(edge, dtype=int) - 1  # Convert to 0-based
        
    if edge.shape[1] != 2:
        raise ValueError('EDGE must be an Mx2 array.')
    if np.max(edge) >= nnode or np.any(edge < 0):
        raise ValueError('Invalid EDGE.')
    
    n = p.shape[0]
    nc = edge.shape[0]
    
    # Choose direction with biggest range as "y-coordinate"
    dxy = np.max(p, axis=0) - np.min(p, axis=0)
    if dxy[0] > dxy[1]:
        # Flip co-ords if x range is bigger
        p = p[:, [1, 0]]
        node = node[:, [1, 0]]
    
    tol = TOL * np.min(dxy)
    
    # Sort test points by y-value
    sort_idx = np.argsort(p[:, 1])
    y = p[sort_idx, 1]
    x = p[sort_idx, 0]
    
    # Main loop
    enum = np.zeros(p.shape[0], dtype=int)
    
    for k in range(nc):  # Loop through edges
        # Nodes in current edge
        n1 = edge[k, 0]
        n2 = edge[k, 1]
        
        # Endpoints - sorted so that [x1,y1] & [x2,y2] has y1<=y2
        y1 = node[n1, 1]
        y2 = node[n2, 1]
        if y1 < y2:
            x1 = node[n1, 0]
            x2 = node[n2, 0]
        else:
            y1, y2 = y2, y1
            x1 = node[n2, 0]
            x2 = node[n1, 0]
        
        # Binary search to find first point with y>=y1 for current edge
        if y[0] >= y1:
            start = 0
        elif y[n-1] < y1:
            start = n
        else:
            lower = 0
            upper = n - 1
            while upper - lower > 1:
                mid = (lower + upper) // 2
                if y[mid] < y1:
                    lower = mid
                else:
                    upper = mid
            start = upper
        
        # Loop through points
        for j in range(start, n):
            Y = y[j]  # Do the array look-up once & make a temp scalar
            if Y <= y2:
                # Check if we're "on" the edge
                X = x[j]
                if abs((y2 - Y) * (x1 - X) - (y1 - Y) * (x2 - X)) < tol:
                    enum[j] = k + 1  # Convert back to 1-based indexing
            else:
                # Due to sorting, no points with >y value need to be checked
                break
    
    # Re-index to undo the sorting
    result = np.zeros(p.shape[0], dtype=int)
    result[sort_idx] = enum
    
    return result