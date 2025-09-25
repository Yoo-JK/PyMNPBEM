import numpy as np

def bradius(p):
    """
    BRADIUS - Minimal radius for spheres enclosing boundary elements.
    
    Parameters:
    -----------
    p : object
        Discretized particle boundary with pos, faces, verts, n attributes
        
    Returns:
    --------
    np.ndarray
        Minimal radius for spheres enclosing boundary elements
    """
    # Allocate array
    r = np.zeros(p.n)
    
    # Triangles and quadrilateral faces
    ind3 = np.where(np.isnan(p.faces[:, 3]))[0]   # Triangular faces
    ind4 = np.where(~np.isnan(p.faces[:, 3]))[0]  # Quadrilateral faces
    
    # Distance function
    def dist(x, y):
        return np.sqrt(np.sum((x - y)**2, axis=1))
    
    # Maximal distance between centroids and triangle vertices
    if len(ind3) > 0:
        for i in range(3):
            vertex_coords = p.verts[p.faces[ind3, i].astype(int)]
            distances = dist(p.pos[ind3], vertex_coords)
            r[ind3] = np.maximum(r[ind3], distances)
    
    # Maximal distance between centroids and quadrilateral vertices  
    if len(ind4) > 0:
        for i in range(4):
            vertex_coords = p.verts[p.faces[ind4, i].astype(int)]
            distances = dist(p.pos[ind4], vertex_coords)
            r[ind4] = np.maximum(r[ind4], distances)
    
    return r