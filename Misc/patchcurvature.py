import numpy as np
from scipy.linalg import lstsq
from scipy.spatial import distance_matrix

def patchcurvature(FV, usethird=False):
    """
    Calculate the principal curvature directions and values of a triangulated mesh.
    
    This function rotates the data so the normal of the current vertex becomes [-1 0 0],
    fits a least-squares quadratic patch to the local neighborhood, then uses the
    eigenvectors and eigenvalues of the Hessian to calculate principal, mean and 
    Gaussian curvature.
    
    Parameters:
    -----------
    FV : dict
        Triangulated mesh with 'vertices' and 'faces' keys
        - vertices: (N, 3) array of vertex coordinates
        - faces: (M, 3) array of face indices (0-based)
    usethird : bool, optional
        Use third order neighbor vertices for smoother but less local fit (default: False)
        
    Returns:
    --------
    tuple
        Cmean : np.ndarray
            Mean curvature
        Cgaussian : np.ndarray  
            Gaussian curvature
        Dir1 : np.ndarray
            XYZ direction of first principal component
        Dir2 : np.ndarray
            XYZ direction of second principal component
        Lambda1 : np.ndarray
            Value of first principal component
        Lambda2 : np.ndarray
            Value of second principal component
    """
    # Number of vertices
    nv = FV['vertices'].shape[0]
    
    # Calculate vertex normals
    N = _patch_normals(FV)
    
    # Calculate rotation matrices for the normals
    M = np.zeros((3, 3, nv))
    Minv = np.zeros((3, 3, nv))
    
    for i in range(nv):
        M[:, :, i], Minv[:, :, i] = _vector_rotation_matrix(N[i, :])
    
    # Get neighbors of all vertices
    Ne = _vertex_neighbors(FV)
    
    # Initialize output arrays
    Lambda1 = np.zeros(nv)
    Lambda2 = np.zeros(nv)
    Dir1 = np.zeros((nv, 3))
    Dir2 = np.zeros((nv, 3))
    
    # Loop through all vertices
    for i in range(nv):
        # Get first and second ring neighbors
        if not usethird:
            # Get unique neighbors of neighbors
            neighbors_i = Ne[i]
            Nce = []
            for neighbor in neighbors_i:
                Nce.extend(Ne[neighbor])
            Nce = np.unique(Nce)
        else:
            # Get first, second and third ring neighbors
            neighbors_i = Ne[i]
            second_ring = []
            for neighbor in neighbors_i:
                second_ring.extend(Ne[neighbor])
            second_ring = np.unique(second_ring)
            
            third_ring = []
            for neighbor in second_ring:
                third_ring.extend(Ne[neighbor])
            Nce = np.unique(third_ring)
        
        Ve = FV['vertices'][Nce, :]
        
        # Rotate to make normal [-1 0 0]
        We = Ve @ Minv[:, :, i]
        f = We[:, 0]
        x = We[:, 1] 
        y = We[:, 2]
        
        # Fit quadratic patch: f(x,y) = ax^2 + by^2 + cxy + dx + ey + f
        if len(x) >= 6:  # Need at least 6 points for 6 parameters
            FM = np.column_stack([x**2, y**2, x*y, x, y, np.ones(len(x))])
            try:
                abcdef, _, _, _ = lstsq(FM, f)
                a, b, c = abcdef[0], abcdef[1], abcdef[2]
            except np.linalg.LinAlgError:
                # Fallback if singular matrix
                a, b, c = 0, 0, 0
        else:
            a, b, c = 0, 0, 0
        
        # Make Hessian matrix elements
        Dxx = 2 * a
        Dxy = c
        Dyy = 2 * b
        
        # Calculate eigenvalues and eigenvectors
        lambda1, lambda2, I1, I2 = _eig2(Dxx, Dxy, Dyy)
        
        # Transform eigenvectors back to 3D
        dir1 = np.array([0, I1[0], I1[1]]) @ M[:, :, i]
        dir2 = np.array([0, I2[0], I2[1]]) @ M[:, :, i]
        
        # Normalize directions
        dir1_norm = np.linalg.norm(dir1)
        dir2_norm = np.linalg.norm(dir2)
        
        if dir1_norm > 0:
            Dir1[i, :] = dir1 / dir1_norm
        if dir2_norm > 0:
            Dir2[i, :] = dir2 / dir2_norm
            
        Lambda1[i] = lambda1
        Lambda2[i] = lambda2
    
    # Calculate mean and Gaussian curvature
    Cmean = (Lambda1 + Lambda2) / 2
    Cgaussian = Lambda1 * Lambda2
    
    return Cmean, Cgaussian, Dir1, Dir2, Lambda1, Lambda2


def _eig2(Dxx, Dxy, Dyy):
    """
    Calculate eigenvalues and eigenvectors of 2x2 matrix.
    
    Matrix: | Dxx  Dxy |
            | Dxy  Dyy |
    """
    # Compute the eigenvectors
    tmp = np.sqrt((Dxx - Dyy)**2 + 4*Dxy**2)
    v2x = 2*Dxy
    v2y = Dyy - Dxx + tmp
    
    # Normalize
    mag = np.sqrt(v2x**2 + v2y**2)
    if mag != 0:
        v2x = v2x / mag
        v2y = v2y / mag
    
    # The eigenvectors are orthogonal
    v1x = -v2y
    v1y = v2x
    
    # Compute the eigenvalues
    mu1 = abs(0.5 * (Dxx + Dyy + tmp))
    mu2 = abs(0.5 * (Dxx + Dyy - tmp))
    
    # Sort eigen values by absolute value abs(Lambda1) < abs(Lambda2)
    if mu1 < mu2:
        Lambda1 = mu1
        Lambda2 = mu2
        I1 = np.array([v1x, v1y])
        I2 = np.array([v2x, v2y])
    else:
        Lambda1 = mu2
        Lambda2 = mu1
        I1 = np.array([v2x, v2y])
        I2 = np.array([v1x, v1y])
    
    return Lambda1, Lambda2, I1, I2


def _patch_normals(FV):
    """
    Calculate normals of a triangulated mesh.
    
    This function calculates the normals of all faces, then calculates 
    vertex normals from face normals weighted by the face angles.
    """
    vertices = FV['vertices']
    faces = FV['faces']
    
    # Get all edge vectors
    e1 = vertices[faces[:, 0], :] - vertices[faces[:, 1], :]
    e2 = vertices[faces[:, 1], :] - vertices[faces[:, 2], :]  
    e3 = vertices[faces[:, 2], :] - vertices[faces[:, 0], :]
    
    # Normalize edge vectors
    e1_norm = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    e2_norm = e2 / np.linalg.norm(e2, axis=1, keepdims=True)
    e3_norm = e3 / np.linalg.norm(e3, axis=1, keepdims=True)
    
    # Calculate angle of face seen from vertices
    angles = np.zeros((len(faces), 3))
    angles[:, 0] = np.arccos(np.clip(np.sum(e1_norm * -e3_norm, axis=1), -1, 1))
    angles[:, 1] = np.arccos(np.clip(np.sum(e2_norm * -e1_norm, axis=1), -1, 1))
    angles[:, 2] = np.arccos(np.clip(np.sum(e3_norm * -e2_norm, axis=1), -1, 1))
    
    # Calculate normal of face
    normals = np.cross(e1, e3)
    
    # Calculate vertex normals
    vertex_normals = np.zeros_like(vertices)
    for i, face in enumerate(faces):
        vertex_normals[face[0], :] += normals[i, :] * angles[i, 0]
        vertex_normals[face[1], :] += normals[i, :] * angles[i, 1]
        vertex_normals[face[2], :] += normals[i, :] * angles[i, 2]
    
    # Normalize vertex normals
    v_norm = np.linalg.norm(vertex_normals, axis=1, keepdims=True) + np.finfo(float).eps
    vertex_normals = vertex_normals / v_norm
    
    return vertex_normals


def _vector_rotation_matrix(v):
    """
    Create rotation matrix to align vector v with [-1, 0, 0].
    """
    v = v / np.linalg.norm(v)
    
    # Create orthogonal vectors
    k = np.random.rand(3)
    k = k / np.linalg.norm(k)
    
    # Create first orthogonal vector
    l = np.cross(k, v)
    l = l / np.linalg.norm(l)
    
    # Create second orthogonal vector
    k = np.cross(l, v)
    k = k / np.linalg.norm(k)
    
    Minv = np.column_stack([v, l, k])
    M = np.linalg.inv(Minv)
    
    return M, Minv


def _vertex_neighbors(FV):
    """
    Find all neighbors of each vertex in a triangulated mesh.
    """
    vertices = FV['vertices']
    faces = FV['faces']
    
    nv = len(vertices)
    neighbors = [[] for _ in range(nv)]
    
    # Build neighbor lists
    for face in faces:
        # Add neighbors for each vertex of the face
        neighbors[face[0]].extend([face[1], face[2]])
        neighbors[face[1]].extend([face[2], face[0]])
        neighbors[face[2]].extend([face[0], face[1]])
    
    # Remove duplicates and sort
    for i in range(nv):
        neighbors[i] = list(set(neighbors[i]))
    
    return neighbors


# Example usage and testing
if __name__ == "__main__":
    # Create a simple test mesh (tetrahedron)
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0], 
        [0.5, np.sqrt(3)/2, 0],
        [0.5, np.sqrt(3)/6, np.sqrt(2/3)]
    ])
    
    faces = np.array([
        [0, 1, 2],
        [0, 3, 1],
        [1, 3, 2],
        [2, 3, 0]
    ])
    
    FV = {'vertices': vertices, 'faces': faces}
    
    # Calculate curvatures
    Cmean, Cgaussian, Dir1, Dir2, Lambda1, Lambda2 = patchcurvature(FV)
    
    print(f"Mean curvature range: {np.min(Cmean):.4f} to {np.max(Cmean):.4f}")
    print(f"Gaussian curvature range: {np.min(Cgaussian):.4f} to {np.max(Cgaussian):.4f}")
    print(f"Principal curvature 1 range: {np.min(Lambda1):.4f} to {np.max(Lambda1):.4f}")
    print(f"Principal curvature 2 range: {np.min(Lambda2):.4f} to {np.max(Lambda2):.4f}")