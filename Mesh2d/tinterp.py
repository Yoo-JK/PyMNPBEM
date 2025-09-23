import numpy as np
from scipy.spatial.distance import cdist


def tinterp(p, t, f, pi, i):
    """
    TINTERP: Triangle based linear interpolation.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        Nodal XY coordinates, [x1,y1; x2,y2; etc]
    t : ndarray, shape (M, 3)
        Triangles as indices, [n11,n12,n13; n21,n22,n23; etc]
    f : ndarray, shape (N,) or (N, K)
        Function vector(s), f(x,y). Can be multi-dimensional.
    pi : ndarray, shape (J, 2)
        Interpolation points
    i : ndarray, shape (J,)
        Triangle indices for each interpolation point.
        NaN or negative values indicate points outside triangulation.
        
    Returns:
    --------
    fi : ndarray, shape (J,) or (J, K)
        Interpolated function values at pi
        
    Notes:
    ------
    Performs nearest-neighbour extrapolation for points outside the
    triangulation using barycentric coordinates for interior points.
    
    The interpolation uses barycentric coordinates within triangles:
    For a point P inside triangle with vertices A, B, C, the interpolated
    value is: f(P) = w1*f(A) + w2*f(B) + w3*f(C)
    where w1, w2, w3 are the barycentric weights.
    
    Darren Engwirda - 2005-2007
    Python conversion - 2025
    """
    
    # Convert inputs to numpy arrays
    p = np.asarray(p)
    t = np.asarray(t, dtype=int)
    f = np.asarray(f)
    pi = np.asarray(pi)
    i = np.asarray(i)
    
    # Handle multi-dimensional function values
    if f.ndim == 1:
        f = f.reshape(-1, 1)
        single_function = True
    else:
        single_function = False
    
    # Input validation
    if p.shape[1] != 2:
        raise ValueError("p must have 2 columns (x, y coordinates)")
    if t.shape[1] != 3:
        raise ValueError("t must have 3 columns (triangle indices)")
    if f.shape[0] != p.shape[0]:
        raise ValueError("f must have same number of rows as p")
    if pi.shape[1] != 2:
        raise ValueError("pi must have 2 columns (x, y coordinates)")
    if len(i) != pi.shape[0]:
        raise ValueError("i must have same length as number of interpolation points")
    
    # Initialize output array
    fi = np.zeros((pi.shape[0], f.shape[1]))
    
    # Identify points outside convex hull
    out = (i < 0) | (i >= t.shape[0]) | np.isnan(i)
    
    # Handle points outside triangulation with nearest neighbor extrapolation
    if np.any(out):
        # Find nearest nodes for outside points
        out_points = pi[out, :]
        distances = cdist(out_points, p)
        nearest_indices = np.argmin(distances, axis=1)
        fi[out, :] = f[nearest_indices, :]
    
    # Handle points inside triangulation
    inside = ~out
    if np.any(inside):
        # Get interior points and their triangle indices
        pin = pi[inside, :]
        i_in = i[inside].astype(int)
        tin = t[i_in, :]
        
        # Extract triangle vertices
        t1 = tin[:, 0]  # First vertex indices
        t2 = tin[:, 1]  # Second vertex indices  
        t3 = tin[:, 2]  # Third vertex indices
        
        # Calculate barycentric coordinates using area ratios
        # For point P and triangle ABC, barycentric coordinates are:
        # w1 = Area(PBC) / Area(ABC)
        # w2 = Area(APC) / Area(ABC) 
        # w3 = Area(ABP) / Area(ABC)
        
        # Vectors from vertices to interpolation points
        dp1 = pin - p[t1, :]  # P - A
        dp2 = pin - p[t2, :]  # P - B
        dp3 = pin - p[t3, :]  # P - C
        
        # Calculate sub-triangle areas using cross product
        # Area = 0.5 * |cross_product|, but we only need ratios so skip 0.5
        A1 = np.abs(dp3[:, 0] * dp2[:, 1] - dp3[:, 1] * dp2[:, 0])  # Area opposite to vertex 1
        A2 = np.abs(dp1[:, 0] * dp3[:, 1] - dp1[:, 1] * dp3[:, 0])  # Area opposite to vertex 2  
        A3 = np.abs(dp1[:, 0] * dp2[:, 1] - dp1[:, 1] * dp2[:, 0])  # Area opposite to vertex 3
        
        # Total area for normalization
        A_total = A1 + A2 + A3
        
        # Avoid division by zero
        A_total = np.maximum(A_total, np.finfo(float).eps)
        
        # Compute barycentric weights
        w1 = A1 / A_total
        w2 = A2 / A_total
        w3 = A3 / A_total
        
        # Linear interpolation using barycentric coordinates
        # fi = w1 * f(t1) + w2 * f(t2) + w3 * f(t3)
        for k in range(f.shape[1]):
            fi[inside, k] = (w1 * f[t1, k] + 
                            w2 * f[t2, k] + 
                            w3 * f[t3, k])
    
    # Return single column if input was 1D
    if single_function:
        return fi.ravel()
    else:
        return fi


def tinterp_gradient(p, t, f, pi, i):
    """
    Triangle based linear interpolation with gradient computation.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        Nodal coordinates
    t : ndarray, shape (M, 3)
        Triangle connectivity
    f : ndarray, shape (N,) or (N, K)
        Function values at nodes
    pi : ndarray, shape (J, 2)
        Interpolation points
    i : ndarray, shape (J,)
        Triangle indices
        
    Returns:
    --------
    fi : ndarray, shape (J,) or (J, K)
        Interpolated values
    grad_fi : ndarray, shape (J, 2) or (J, 2, K)
        Interpolated gradients [df/dx, df/dy]
    """
    
    # Get interpolated values
    fi = tinterp(p, t, f, pi, i)
    
    # Convert to ensure proper dimensions
    p = np.asarray(p)
    t = np.asarray(t, dtype=int)
    f = np.asarray(f)
    pi = np.asarray(pi)
    i = np.asarray(i)
    
    if f.ndim == 1:
        f = f.reshape(-1, 1)
        single_function = True
    else:
        single_function = False
    
    # Initialize gradient array
    if single_function:
        grad_fi = np.zeros((pi.shape[0], 2))
    else:
        grad_fi = np.zeros((pi.shape[0], 2, f.shape[1]))
    
    # Only compute gradients for interior points
    inside = (i >= 0) & (i < t.shape[0]) & ~np.isnan(i)
    
    if np.any(inside):
        i_in = i[inside].astype(int)
        tin = t[i_in, :]
        
        # Triangle vertices
        p1 = p[tin[:, 0], :]
        p2 = p[tin[:, 1], :]
        p3 = p[tin[:, 2], :]
        
        # Calculate gradients using linear triangle elements
        # For linear triangle: f(x,y) = a + bx + cy
        # Gradient = [b, c] is constant within each triangle
        
        for k in range(f.shape[1]):
            f1 = f[tin[:, 0], k]
            f2 = f[tin[:, 1], k]
            f3 = f[tin[:, 2], k]
            
            # Calculate triangle areas and gradients
            det = ((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - 
                   (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1]))
            
            # Avoid division by zero
            det = np.where(np.abs(det) < np.finfo(float).eps, 
                          np.finfo(float).eps, det)
            
            # Gradient components
            dfdx = ((f2 - f1) * (p3[:, 1] - p1[:, 1]) - 
                    (f3 - f1) * (p2[:, 1] - p1[:, 1])) / det
            
            dfdy = ((f3 - f1) * (p2[:, 0] - p1[:, 0]) - 
                    (f2 - f1) * (p3[:, 0] - p1[:, 0])) / det
            
            if single_function:
                grad_fi[inside, 0] = dfdx
                grad_fi[inside, 1] = dfdy
            else:
                grad_fi[inside, 0, k] = dfdx
                grad_fi[inside, 1, k] = dfdy
    
    return fi, grad_fi


def tinterp_vectorized(p, t, f, pi, triangle_indices=None):
    """
    Vectorized triangle interpolation without pre-computed triangle indices.
    
    This version finds the enclosing triangle for each point automatically.
    """
    from matplotlib.path import Path
    
    p = np.asarray(p)
    t = np.asarray(t, dtype=int)
    f = np.asarray(f)
    pi = np.asarray(pi)
    
    if f.ndim == 1:
        f = f.reshape(-1, 1)
        single_function = True
    else:
        single_function = False
    
    # Initialize output and triangle index arrays
    fi = np.zeros((pi.shape[0], f.shape[1]))
    i = np.full(pi.shape[0], -1, dtype=int)
    
    # Find enclosing triangle for each point
    for tri_idx in range(t.shape[0]):
        # Get triangle vertices
        tri_verts = p[t[tri_idx, :], :]
        
        # Create path for this triangle
        triangle_path = Path(tri_verts)
        
        # Find points inside this triangle
        inside_mask = triangle_path.contains_points(pi)
        
        # Update triangle indices
        i[inside_mask] = tri_idx
        
        # Early termination if all points are assigned
        if np.all(i >= 0):
            break
    
    # Interpolate using found triangle indices
    return tinterp(p, t, f, pi, i)


def dsearch_replacement(px, py, t, xi, yi):
    """
    Replacement for MATLAB's dsearch function.
    
    Find nearest node for each query point.
    """
    # Stack points
    p = np.column_stack([px, py])
    pi = np.column_stack([xi, yi])
    
    # Calculate distances
    distances = cdist(pi, p)
    nearest_indices = np.argmin(distances, axis=1)
    
    return nearest_indices


# Example usage and testing
def tinterp_example():
    """
    Example demonstrating triangle interpolation usage.
    """
    # Create a simple triangular mesh
    p = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0],
        [1.5, 0.5],
        [0.0, 1.0]
    ])
    
    t = np.array([
        [0, 1, 2],
        [1, 3, 2],
        [0, 2, 4]
    ])
    
    # Define a test function: f(x,y) = x^2 + y^2
    f = p[:, 0]**2 + p[:, 1]**2
    
    # Interpolation points
    pi = np.array([
        [0.25, 0.25],  # Inside first triangle
        [0.75, 0.25],  # Inside second triangle  
        [2.0, 2.0],    # Outside triangulation
        [0.1, 0.8]     # Inside third triangle
    ])
    
    # Find triangle indices (simplified - in practice use tsearch)
    i = np.array([0, 1, -1, 2])  # -1 indicates outside
    
    # Perform interpolation
    fi = tinterp(p, t, f, pi, i)
    
    print("Interpolation example:")
    print("Points:", pi)
    print("Triangle indices:", i)
    print("Interpolated values:", fi)
    print("True values:", pi[:, 0]**2 + pi[:, 1]**2)
    
    return p, t, f, pi, fi


def test_tinterp():
    """
    Test function for triangle interpolation.
    """
    # Run example
    p, t, f, pi, fi = tinterp_example()
    
    # Test multi-dimensional function
    f_multi = np.column_stack([f, 2*f, f**0.5])
    fi_multi = tinterp(p, t, f_multi, pi, np.array([0, 1, -1, 2]))
    
    print("\nMulti-dimensional interpolation:")
    print("Shape of interpolated values:", fi_multi.shape)
    print("Values:", fi_multi)
    
    return True