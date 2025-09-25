import numpy as np
from scipy.spatial import KDTree
from matplotlib.path import Path

def distmin3(p, pos, cutoff=None):
    """
    DISTMIN3 - Minimum distance in 3D between particle faces and positions.
    
    Parameters:
    -----------
    p : object
        Particle object with pos, nvec, tvec1, tvec2, faces, verts attributes
    pos : array_like
        Positions (N x 3 array)
    cutoff : float, optional
        Compute distances correctly only for dmin < cutoff (default: inf)
        
    Returns:
    --------
    tuple
        dmin : np.ndarray
            Minimum distance between particle faces and positions
        ind : np.ndarray
            Index to nearest neighbor faces
    """
    pos = np.asarray(pos)
    
    # Integration points for particle boundary
    pos2, _, iface2 = p.quad()  # Assuming quad method exists
    
    # Find integration points closest to pos and convert to face index
    try:
        # Use KDTree as substitute for MATLAB's knnsearch
        tree = KDTree(pos2)
        _, indices = tree.query(pos)
        ind = iface2[indices]
    except:
        # Fallback: use distance calculation
        distances = np.linalg.norm(pos[:, np.newaxis] - p.pos[np.newaxis, :], axis=2)
        ind = np.argmin(distances, axis=1)
    
    # Distance between centroids and positions along normal direction
    # Sign of distance: positive if point above face element, negative otherwise
    dmin = np.sum((pos - p.pos[ind]) * p.nvec[ind], axis=1)
    
    if cutoff is not None and cutoff == 0:
        return dmin, ind
    
    if cutoff is None:
        cutoff = np.inf
    
    # Loop over positions where refinement is needed
    for i in np.where(np.abs(dmin) <= cutoff)[0]:
        # Centroid position
        pos0 = p.pos[ind[i]]
        
        # Project position onto plane perpendicular to nvec
        x = np.dot(pos[i] - pos0, p.tvec1[ind[i]])
        y = np.dot(pos[i] - pos0, p.tvec2[ind[i]])
        
        # Get face vertices
        face_indices = p.faces[ind[i]]
        valid_indices = face_indices[~np.isnan(face_indices)]
        verts = p.verts[valid_indices.astype(int)]
        
        # Project vertices onto tangent plane
        verts_rel = verts - pos0
        xv = np.dot(verts_rel, p.tvec1[ind[i]])
        yv = np.dot(verts_rel, p.tvec2[ind[i]])
        
        # Distance from point to polygon in 2D
        rmin = _p_poly_dist(x, y, xv, yv)
        
        # Add to dmin if point is located outside of polygon
        if rmin > 0:
            dmin[i] = np.sign(dmin[i]) * np.sqrt(dmin[i]**2 + rmin**2)
    
    return dmin, ind


def _p_poly_dist(x, y, xv, yv):
    """
    Distance from point to polygon whose vertices are specified by vectors xv and yv.
    
    Parameters:
    -----------
    x, y : float
        Point coordinates
    xv, yv : array_like
        Polygon vertex coordinates
        
    Returns:
    --------
    float
        Distance from point to polygon (positive if outside, negative if inside)
    """
    xv = np.asarray(xv).flatten()
    yv = np.asarray(yv).flatten()
    
    # Close polygon if not already closed
    if len(xv) > 0 and (xv[0] != xv[-1] or yv[0] != yv[-1]):
        xv = np.append(xv, xv[0])
        yv = np.append(yv, yv[0])
    
    if len(xv) < 2:
        return np.inf
    
    # Linear parameters of segments connecting vertices
    dx = np.diff(xv)
    dy = np.diff(yv)
    A = -dy
    B = dx
    C = yv[1:] * xv[:-1] - xv[1:] * yv[:-1]
    
    # Find projection of point (x,y) on each segment
    AB_inv = 1.0 / (A**2 + B**2)
    vv = A*x + B*y + C
    xp = x - A * AB_inv * vv
    yp = y - B * AB_inv * vv
    
    # Check if projected point is inside each segment
    idx_x = ((xp >= np.minimum(xv[:-1], xv[1:])) & 
             (xp <= np.maximum(xv[:-1], xv[1:])))
    idx_y = ((yp >= np.minimum(yv[:-1], yv[1:])) & 
             (yp <= np.maximum(yv[:-1], yv[1:])))
    idx = idx_x & idx_y
    
    # Distance from point to vertices
    dv = np.sqrt((xv[:-1] - x)**2 + (yv[:-1] - y)**2)
    
    if not np.any(idx):
        # All projections are outside polygon segments
        d = np.min(dv)
    else:
        # Distance from point to projections on segments
        dp = np.sqrt((xp[idx] - x)**2 + (yp[idx] - y)**2)
        d = min(np.min(dv), np.min(dp))
    
    # Check if point is inside polygon using matplotlib's Path
    if len(xv) > 2:
        polygon_path = Path(np.column_stack([xv, yv]))
        if polygon_path.contains_point([x, y]):
            d = -d
    
    return d