import numpy as np
from scipy.spatial import Delaunay
import warnings


def mydelaunayn(p):
    """
    My version of the MATLAB delaunayn function that attempts to deal with
    some of the compatibility and robustness problems.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        Array of points to triangulate
        
    Returns:
    --------
    t : ndarray, shape (M, 3)
        Array of triangles as indices into p
        
    Notes:
    ------
    Translates to origin and scales the min xy range onto [-1,1].
    This is absolutely critical to avoid precision issues for large problems!
    
    Darren Engwirda - 2007
    Python conversion - 2025
    """
    
    if p.size == 0:
        return np.array([]).reshape(0, 3)
    
    # Store original points for reference
    p_orig = p.copy()
    
    # Translate to the origin and scale the min xy range onto [-1,1]
    # This is absolutely critical to avoid precision issues for large problems!
    maxxy = np.max(p, axis=0)
    minxy = np.min(p, axis=0)
    
    # Translate to origin
    center = 0.5 * (minxy + maxxy)
    p = p - center
    
    # Scale to [-1, 1]
    scale_factor = 0.5 * np.min(maxxy - minxy)
    if scale_factor > 0:
        p = p / scale_factor
    
    try:
        # Use scipy's Delaunay triangulation
        tri = Delaunay(p, qhull_options='Qt Qbb Qc')
        t = tri.simplices
        
        # Remove degenerate triangles (zero area)
        t = remove_degenerate_triangles(p, t)
        
    except Exception as e:
        # If standard triangulation fails, try with more robust options
        try:
            # Add joggling to handle precision issues
            tri = Delaunay(p, qhull_options='Qt Qbb Qc QJ')
            t = tri.simplices
            
            # Remove degenerate triangles
            t = remove_degenerate_triangles(p, t)
            
        except Exception as e2:
            warnings.warn(f"Delaunay triangulation failed: {e2}")
            return np.array([]).reshape(0, 3)
    
    return t


def remove_degenerate_triangles(p, t):
    """
    Remove triangles with zero or very small area.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        Points array
    t : ndarray, shape (M, 3)
        Triangle indices
        
    Returns:
    --------
    t : ndarray
        Filtered triangles
    """
    if t.size == 0:
        return t
    
    # Calculate triangle areas using cross product
    d12 = p[t[:, 1], :] - p[t[:, 0], :]
    d13 = p[t[:, 2], :] - p[t[:, 0], :]
    
    # Area = 0.5 * |cross product|
    areas = np.abs(d12[:, 0] * d13[:, 1] - d12[:, 1] * d13[:, 0])
    
    # Set threshold for minimum area (similar to MATLAB's eps^(4/5))
    seps = np.finfo(float).eps**(4/5) * np.max(np.abs(p))
    
    # Keep only triangles with sufficient area
    valid_triangles = areas > seps
    t = t[valid_triangles, :]
    
    return t


def delaunayn_optimized(x, options=None):
    """
    2D optimised Qhull call (alternative implementation).
    
    This is based on the commented MATLAB code for faster implementation.
    
    Parameters:
    -----------
    x : ndarray, shape (N, 2)
        Points to triangulate
    options : list of str, optional
        Qhull options
        
    Returns:
    --------
    t : ndarray, shape (M, 3)
        Triangle indices
    """
    
    if x.size == 0:
        return np.array([]).reshape(0, 3)
    
    m, n = x.shape
    
    if m < n + 1:
        raise ValueError('Not enough unique points to do tessellation.')
    
    if np.any(np.isinf(x)) or np.any(np.isnan(x)):
        raise ValueError('Data containing Inf or NaN cannot be tessellated.')
    
    if m == n + 1:
        return np.arange(n + 1).reshape(1, -1)
    
    # Deal with options
    if n >= 4:
        opt = 'Qt Qbb Qc Qx'
    else:
        opt = 'Qt Qbb Qc'
    
    if options is not None:
        if not isinstance(options, list):
            raise ValueError('OPTIONS should be list of strings.')
        opt = ' '.join(options)
    
    # Call scipy's Delaunay with specified options
    try:
        tri = Delaunay(x, qhull_options=opt)
        t = tri.simplices
    except Exception:
        # Fallback to default options
        tri = Delaunay(x)
        t = tri.simplices
    
    # Try to get rid of zero volume simplices
    seps = np.finfo(float).eps**(4/5) * np.max(np.abs(x))
    
    # Triangle area calculation
    d12 = x[t[:, 1], :] - x[t[:, 0], :]
    d13 = x[t[:, 2], :] - x[t[:, 0], :]
    A = d12[:, 0] * d13[:, 1] - d12[:, 1] * d13[:, 0]
    
    # Keep triangles with sufficient area
    t = t[np.abs(A) > seps, :]
    
    return t


# Alternative function name for compatibility
def mydelaunay(p):
    """Alias for mydelaunayn for backward compatibility."""
    return mydelaunayn(p)