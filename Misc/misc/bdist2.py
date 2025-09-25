import numpy as np

def bdist2(p1, p2):
    """
    BDIST2 - Minimal distance between positions P1 and boundary elements P2.
    
    Parameters:
    -----------
    p1 : object
        Object with pos attribute (positions)
    p2 : object
        Discretized particle boundary with verts and faces
        
    Returns:
    --------
    np.ndarray
        Minimal distance between P1.pos and boundary elements P2
    """
    # Allocate distance array
    d = np.zeros((p1.n, p2.n))
    
    # Triangles and quadrilateral faces
    ind3 = np.where(np.isnan(p2.faces[:, 3]))[0]  # Triangular faces
    ind4 = np.where(~np.isnan(p2.faces[:, 3]))[0]  # Quadrilateral faces
    
    # Minimal distance between points and edges of boundary elements
    if len(ind3) > 0:
        d[:, ind3] = _point_edge_dist(p1.pos, p2.verts, p2.faces[ind3, :3])
    if len(ind4) > 0:
        d[:, ind4] = _point_edge_dist(p1.pos, p2.verts, p2.faces[ind4, :4])
    
    # Compare with points inside boundary elements
    if len(ind3) > 0:
        d[:, ind3] = np.minimum(d[:, ind3], 
                               _point_triangle_dist(p1.pos, p2.verts, p2.faces[ind3, :3]))
    
    if len(ind4) > 0:
        # Split quadrilaterals into two triangles
        tri1 = _point_triangle_dist(p1.pos, p2.verts, p2.faces[ind4, [0, 1, 2]])
        tri2 = _point_triangle_dist(p1.pos, p2.verts, p2.faces[ind4, [2, 3, 0]])
        d[:, ind4] = np.minimum(d[:, ind4], np.minimum(tri1, tri2))
    
    return d


def _point_edge_dist(pos, verts, faces):
    """
    Minimal distance between points and boundary edges.
    
    The edge boundary is parameterized through g: r = Q + lambda * a
    """
    n_pos = pos.shape[0]
    n_faces = faces.shape[0]
    
    if n_faces == 0:
        return np.full((n_pos, n_faces), np.inf)
    
    # Initialize distance array
    d = np.full((n_pos, n_faces), np.inf)
    
    # Close faces by appending first vertex at end
    faces_closed = np.column_stack([faces, faces[:, 0]])
    
    # Loop over face boundaries
    for i in range(1, faces_closed.shape[1]):
        # First and second vertex
        valid_v1 = ~np.isnan(faces_closed[:, i-1])
        valid_v2 = ~np.isnan(faces_closed[:, i])
        valid = valid_v1 & valid_v2
        
        if not np.any(valid):
            continue
            
        v1_indices = faces_closed[valid, i-1].astype(int)
        v2_indices = faces_closed[valid, i].astype(int)
        
        v1 = verts[v1_indices]
        v2 = verts[v2_indices]
        
        # Difference vector
        a = v2 - v1
        
        # Parameter for minimal distance
        pos_dot_a = pos @ a.T  # (n_pos, n_valid)
        v1_dot_a = np.sum(v1 * a, axis=1)  # (n_valid,)
        a_dot_a = np.sum(a * a, axis=1)  # (n_valid,)
        
        # Avoid division by zero
        a_dot_a = np.where(a_dot_a == 0, 1e-12, a_dot_a)
        
        lambda_param = (pos_dot_a - v1_dot_a[np.newaxis, :]) / a_dot_a[np.newaxis, :]
        
        # Clamp lambda to [0, 1]
        lambda_param = np.clip(lambda_param, 0, 1)
        
        # Distance calculation
        from .pdist2 import pdist2  # Import local pdist2 function
        pos_v1_dist = pdist2(pos, v1)**2
        
        # Compute minimum distance to line segments
        term1 = lambda_param**2 * a_dot_a[np.newaxis, :]
        term2 = 2 * lambda_param * (pos_dot_a - v1_dot_a[np.newaxis, :])
        
        segment_dist = np.sqrt(pos_v1_dist + term1 - term2)
        
        # Update distances for valid faces
        d[:, valid] = np.minimum(d[:, valid], segment_dist)
    
    return d


def _point_triangle_dist(pos, verts, faces):
    """
    Normal distance between points and triangle plane.
    
    If the crossing point lies outside the triangle, distance is set to infinity.
    """
    n_pos = pos.shape[0]
    n_faces = faces.shape[0]
    
    if n_faces == 0:
        return np.full((n_pos, n_faces), np.inf)
    
    # Initialize distance array
    d = np.full((n_pos, n_faces), np.inf)
    
    # Triangle vertices
    v1 = verts[faces[:, 0].astype(int)]
    v2 = verts[faces[:, 1].astype(int)]
    v3 = verts[faces[:, 2].astype(int)]
    
    # Vectors of plane
    a1 = v2 - v1
    a2 = v3 - v1
    
    # Inner products
    pos_dot_a1 = pos @ a1.T
    pos_dot_a2 = pos @ a2.T
    v1_dot_a1 = np.sum(v1 * a1, axis=1)
    v1_dot_a2 = np.sum(v1 * a2, axis=1)
    
    in1 = pos_dot_a1 - v1_dot_a1[np.newaxis, :]
    in2 = pos_dot_a2 - v1_dot_a2[np.newaxis, :]
    
    # Inner products of plane vectors
    a11 = np.sum(a1 * a1, axis=1)
    a12 = np.sum(a1 * a2, axis=1)
    a22 = np.sum(a2 * a2, axis=1)
    
    # Determinant
    det = a11 * a22 - a12**2
    
    # Avoid division by zero
    det = np.where(det == 0, 1e-12, det)
    
    # Crossing point parameters
    mu1 = (in1 * a22[np.newaxis, :] - in2 * a12[np.newaxis, :]) / det[np.newaxis, :]
    mu2 = (in2 * a11[np.newaxis, :] - in1 * a12[np.newaxis, :]) / det[np.newaxis, :]
    
    # Find elements inside triangles
    inside = ((mu1 >= 0) & (mu1 <= 1) & 
              (mu2 >= 0) & (mu2 <= 1) & 
              (mu1 + mu2 <= 1))
    
    # Normal vector of plane
    nvec = np.cross(a1, a2)
    nvec_norm = np.linalg.norm(nvec, axis=1)
    nvec_norm = np.where(nvec_norm == 0, 1, nvec_norm)
    nvec = nvec / nvec_norm[:, np.newaxis]
    
    # Distance to triangle planes
    pos_dot_nvec = pos @ nvec.T
    v1_dot_nvec = np.sum(v1 * nvec, axis=1)
    
    d = np.abs(pos_dot_nvec - v1_dot_nvec[np.newaxis, :])
    
    # Set distance to infinity if crossing point lies outside triangle
    d[~inside] = np.inf
    
    return d