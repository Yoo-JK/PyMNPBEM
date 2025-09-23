import numpy as np


def inpoly(p, node, edge=None, reltol=1.0e-12):
    """
    Point-in-polygon testing.
    
    Determine whether a series of points lie within the bounds of a polygon
    in the 2D plane. General non-convex, multiply-connected polygonal
    regions can be handled.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        The points to be tested as an array [x1 y1; x2 y2; etc]
    node : ndarray, shape (M, 2)
        The vertices of the polygon as an array [X1 Y1; X2 Y2; etc]
    edge : ndarray, shape (M, 2), optional
        Array of polygon edges, specified as connections between vertices in NODE
    reltol : float, optional
        Relative tolerance for edge detection (default: 1.0e-12)
        
    Returns:
    --------
    cn : ndarray, shape (N,), dtype=bool
        Logical array with cn[i] = True if p[i,:] lies within the region
        or on an edge
    on : ndarray, shape (N,), dtype=bool
        Logical array with on[i] = True if p[i,:] lies on a polygon edge
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
    
    # Polygon bounding-box
    dxy = np.max(node, axis=0) - np.min(node, axis=0)
    tol = reltol * np.min(dxy)
    if tol == 0.0:
        tol = reltol
    
    # Sort test points by y-value
    sort_idx = np.argsort(p[:, 1])
    y = p[sort_idx, 1]
    x = p[sort_idx, 0]
    
    # Main loop
    cn = np.zeros(n, dtype=bool)  # Crossing number test using logical flip
    on = np.zeros(n, dtype=bool)
    
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
        
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        
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
                X = x[j]  # Do the array look-up once & make a temp scalar
                if X >= xmin:
                    if X <= xmax:
                        # Check if we're "on" the edge
                        on[j] = on[j] or (abs((y2-Y)*(x1-X)-(y1-Y)*(x2-X)) <= tol)
                        
                        # Do the actual intersection test
                        if (Y < y2) and ((y2-y1)*(X-x1) < (Y-y1)*(x2-x1)):
                            cn[j] = not cn[j]
                elif Y < y2:  # Deal with points exactly at vertices
                    # Has to cross edge
                    cn[j] = not cn[j]
            else:
                # Due to sorting, no points with >y value need to be checked
                break
    
    # Re-index to undo the sorting
    cn_result = np.zeros(n, dtype=bool)
    on_result = np.zeros(n, dtype=bool)
    cn_result[sort_idx] = cn | on
    on_result[sort_idx] = on
    
    return cn_result, on_result