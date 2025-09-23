import numpy as np
from matplotlib.path import Path


def mytsearch(x, y, t, xi, yi, i=None):
    """
    MYTSEARCH: Find the enclosing triangle for points in a 2D plane.
    
    Parameters:
    -----------
    x, y : array_like
        Coordinates of triangulation vertices
    t : ndarray, shape (M, 3)
        Triangle connectivity matrix
    xi, yi : array_like
        Query points coordinates
    i : array_like, optional
        Initial guess for triangle indices
        
    Returns:
    --------
    i : ndarray
        Indices of triangles enclosing the points in [XI,YI].
        Points outside the triangulation are assigned NaN.
        
    Notes:
    ------
    The triangulation T of [X,Y] must be convex. Points lying outside the
    triangulation are assigned a NaN index.
    
    IGUESS is an optional initial guess for the indices. A full search is
    done for points with an invalid initial guess.
    
    Darren Engwirda - 2007
    Python conversion - 2025
    """
    
    # Convert inputs to numpy arrays
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    t = np.asarray(t)
    xi = np.asarray(xi).ravel()
    yi = np.asarray(yi).ravel()
    
    # Input validation
    ni = len(xi)
    if len(yi) != ni:
        raise ValueError('xi and yi must have the same length')
    if len(x) != len(y):
        raise ValueError('x and y must have the same length')
    if i is not None:
        i = np.asarray(i).ravel()
        if len(i) != ni:
            raise ValueError('Initial guess i must have same length as xi, yi')
    else:
        i = np.full(ni, -1, dtype=int)
    
    # Translate to the origin and scale the min xy range onto [-1,1]
    # This is absolutely critical to avoid precision issues for large problems!
    all_coords = np.concatenate([x, y])
    maxxy = np.max(all_coords)
    minxy = np.min(all_coords)
    den = 0.5 * (maxxy - minxy)
    
    if den > 0:
        center_x = 0.5 * (minxy + maxxy)
        center_y = 0.5 * (minxy + maxxy)
        
        x = (x - center_x) / den
        y = (y - center_y) / den
        xi = (xi - center_x) / den
        yi = (yi - center_y) / den
    
    # Initialize result array
    result = np.full(ni, np.nan)
    
    # Check initial guess if provided
    needs_search = np.ones(ni, dtype=bool)
    
    if i is not None and len(i) > 0:
        # Find valid initial guesses
        valid_guess = (i >= 0) & (i < len(t)) & ~np.isnan(i)
        
        if np.any(valid_guess):
            k = np.where(valid_guess)[0]
            tri_indices = i[k].astype(int)
            
            # Get triangle vertices
            n1 = t[tri_indices, 0]
            n2 = t[tri_indices, 1] 
            n3 = t[tri_indices, 2]
            
            # Check if points are inside their guessed triangles
            ok = (sameside(x[n1], y[n1], x[n2], y[n2], xi[k], yi[k], x[n3], y[n3]) &
                  sameside(x[n2], y[n2], x[n3], y[n3], xi[k], yi[k], x[n1], y[n1]) &
                  sameside(x[n3], y[n3], x[n1], y[n1], xi[k], yi[k], x[n2], y[n2]))
            
            # Update result for successful guesses
            valid_k = k[ok]
            result[valid_k] = tri_indices[ok]
            needs_search[valid_k] = False
    
    # Do a full search for points that failed initial guess
    search_indices = np.where(needs_search)[0]
    
    if len(search_indices) > 0:
        # Use point-in-polygon test for each triangle
        for tri_idx in range(len(t)):
            # Get triangle vertices
            tri_verts = np.column_stack([
                x[t[tri_idx, :]], 
                y[t[tri_idx, :]]
            ])
            
            # Create path for this triangle
            triangle_path = Path(tri_verts)
            
            # Test which search points are in this triangle
            query_points = np.column_stack([xi[search_indices], yi[search_indices]])
            inside = triangle_path.contains_points(query_points)
            
            # Update result for points found in this triangle
            found_indices = search_indices[inside]
            result[found_indices] = tri_idx
            
            # Remove found points from search list
            search_indices = search_indices[~inside]
            
            # Early exit if all points found
            if len(search_indices) == 0:
                break
    
    # Convert NaN to appropriate format (MATLAB uses NaN for not found)
    result_int = np.full(ni, -1, dtype=int)
    valid_results = ~np.isnan(result)
    result_int[valid_results] = result[valid_results].astype(int)
    result_int[~valid_results] = -1  # Use -1 to indicate not found
    
    return result_int


def sameside(xa, ya, xb, yb, x1, y1, x2, y2):
    """
    Test if [x1(i),y1(i)] and [x2(i),y2(i)] lie on the same side of the line AB(i).
    
    Parameters:
    -----------
    xa, ya, xb, yb : array_like
        Coordinates defining line AB
    x1, y1, x2, y2 : array_like
        Coordinates of points to test
        
    Returns:
    --------
    i : ndarray of bool
        True if points lie on the same side of line AB
    """
    
    # Convert to arrays
    xa, ya, xb, yb = np.broadcast_arrays(xa, ya, xb, yb)
    x1, y1, x2, y2 = np.broadcast_arrays(x1, y1, x2, y2)
    
    # Calculate line direction
    dx = xb - xa
    dy = yb - ya
    
    # Calculate cross products to determine which side of line each point is on
    a1 = (x1 - xa) * dy - (y1 - ya) * dx
    a2 = (x2 - xa) * dy - (y2 - ya) * dx
    
    # If sign(a1) = sign(a2), the points lie on the same side
    # Points on the line (a1=0 or a2=0) are considered on the same side
    same_side = (a1 * a2) >= 0.0
    
    return same_side


def mytsearch_vectorized(x, y, t, xi, yi, i=None):
    """
    Vectorized version of mytsearch for better performance with large datasets.
    
    This version processes all query points simultaneously for each triangle,
    which can be more efficient for large numbers of query points.
    """
    
    # Convert inputs and normalize coordinates (same as above)
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    t = np.asarray(t)
    xi = np.asarray(xi).ravel()
    yi = np.asarray(yi).ravel()
    
    ni = len(xi)
    if i is not None:
        i = np.asarray(i).ravel()
    
    # Normalize coordinates
    all_coords = np.concatenate([x, y])
    maxxy = np.max(all_coords)
    minxy = np.min(all_coords)
    den = 0.5 * (maxxy - minxy)
    
    if den > 0:
        center = 0.5 * (minxy + maxxy)
        x = (x - center) / den
        y = (y - center) / den
        xi = (xi - center) / den
        yi = (yi - center) / den
    
    # Initialize result
    result = np.full(ni, -1, dtype=int)
    
    # For each triangle, test all remaining points
    remaining_mask = np.ones(ni, dtype=bool)
    
    for tri_idx in range(len(t)):
        if not np.any(remaining_mask):
            break
            
        # Get current query points
        current_xi = xi[remaining_mask]
        current_yi = yi[remaining_mask]
        
        # Triangle vertices
        v1_idx, v2_idx, v3_idx = t[tri_idx, :]
        
        # Barycentric coordinate test (more efficient than sameside for many points)
        denom = ((y[v2_idx] - y[v3_idx]) * (x[v1_idx] - x[v3_idx]) + 
                 (x[v3_idx] - x[v2_idx]) * (y[v1_idx] - y[v3_idx]))
        
        if abs(denom) < 1e-12:  # Degenerate triangle
            continue
            
        a = ((y[v2_idx] - y[v3_idx]) * (current_xi - x[v3_idx]) + 
             (x[v3_idx] - x[v2_idx]) * (current_yi - y[v3_idx])) / denom
        
        b = ((y[v3_idx] - y[v1_idx]) * (current_xi - x[v3_idx]) + 
             (x[v1_idx] - x[v3_idx]) * (current_yi - y[v3_idx])) / denom
        
        c = 1 - a - b
        
        # Points inside triangle have all barycentric coordinates >= 0
        inside = (a >= -1e-12) & (b >= -1e-12) & (c >= -1e-12)
        
        if np.any(inside):
            # Update results
            remaining_indices = np.where(remaining_mask)[0]
            found_indices = remaining_indices[inside]
            result[found_indices] = tri_idx
            
            # Remove found points from remaining
            remaining_mask[found_indices] = False
    
    return result