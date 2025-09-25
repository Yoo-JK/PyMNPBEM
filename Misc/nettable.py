import numpy as np

def nettable(faces):
    """
    NETTABLE - Table of connections between vertices.
    
    Parameters:
    -----------
    faces : array_like
        Faces of particle boundary (N x 3 for triangles, N x 4 for quads)
        NaN values indicate triangular faces in mixed meshes
        
    Returns:
    --------
    tuple
        net : np.ndarray
            List of connections between vertices (M x 2)
        inet : np.ndarray
            Face-to-net index mapping
    """
    faces = np.asarray(faces)
    
    # Handle case where faces is 1D or has wrong shape
    if faces.ndim == 1:
        faces = faces.reshape(1, -1)
    
    net = []
    inet = []
    
    # Triangular faces (where 4th column is NaN or doesn't exist)
    if faces.shape[1] == 3:
        # All triangular faces
        i3 = np.arange(faces.shape[0])
    else:
        # Mixed faces - find triangular ones (4th column is NaN)
        i3 = np.where(np.isnan(faces[:, 3]))[0]
    
    if len(i3) > 0:
        # Connections between vertices for triangular faces
        tri_net = np.vstack([
            faces[i3, [0, 1]],  # edge 1-2
            faces[i3, [1, 2]],  # edge 2-3
            faces[i3, [2, 0]]   # edge 3-1
        ])
        
        # Remove any rows with NaN values
        valid_rows = ~np.any(np.isnan(tri_net), axis=1)
        tri_net = tri_net[valid_rows]
        
        # Convert to integer indices
        tri_net = tri_net.astype(int)
        
        # Pointer to corresponding faces (repeated 3 times for each triangle)
        tri_inet = np.tile(i3, 3)
        
        net.append(tri_net)
        inet.append(tri_inet)
    
    # Quadrilateral faces (where 4th column is not NaN)
    if faces.shape[1] >= 4:
        i4 = np.where(~np.isnan(faces[:, 3]))[0]
        
        if len(i4) > 0:
            # Connections between vertices for quadrilateral faces
            quad_net = np.vstack([
                faces[i4, [0, 1]],  # edge 1-2
                faces[i4, [1, 2]],  # edge 2-3
                faces[i4, [2, 3]],  # edge 3-4
                faces[i4, [3, 0]]   # edge 4-1
            ])
            
            # Convert to integer indices
            quad_net = quad_net.astype(int)
            
            # Pointer to corresponding faces (repeated 4 times for each quad)
            quad_inet = np.tile(i4, 4)
            
            net.append(quad_net)
            inet.append(quad_inet)
    
    # Combine results
    if net:
        net = np.vstack(net)
        inet = np.concatenate(inet)
    else:
        net = np.empty((0, 2), dtype=int)
        inet = np.empty(0, dtype=int)
    
    return net, inet