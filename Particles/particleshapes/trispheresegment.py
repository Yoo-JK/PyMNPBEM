import numpy as np
from typing import Union, Any


def trispheresegment(phi, theta, *args, **kwargs):
    """
    Discretized surface of sphere.
    
    Usage:
        p = trispheresegment(phi, theta, diameter, **kwargs)
        p = trispheresegment(phi, theta, diameter, triangles=True, **kwargs)
    
    Parameters:
    -----------
    phi : array-like
        Azimuthal angles
    theta : array-like
        Polar angles
    diameter : float, optional
        Diameter of sphere (default: 1)
    triangles : bool, optional
        Use triangles rather than quadrilaterals (default: False)
    **kwargs : dict
        Additional arguments to be passed to particle constructor
        
    Returns:
    --------
    particle
        Discretized particle surface
    """
    
    # Extract diameter
    if args and isinstance(args[0], (int, float)):
        diameter = args[0]
        remaining_args = args[1:]
    else:
        diameter = 1
        remaining_args = args
    
    # Extract triangles option
    triangles = kwargs.pop('triangles', False)
    
    # Check for 'triangles' string in remaining args (MATLAB style)
    if remaining_args and isinstance(remaining_args[0], str) and remaining_args[0] == 'triangles':
        triangles = True
        remaining_args = remaining_args[1:]
    
    # Particle surface
    # Grid of PHI and THETA values
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    
    # Convert to Cartesian coordinates
    x = diameter / 2 * np.sin(theta_grid) * np.cos(phi_grid)
    y = diameter / 2 * np.sin(theta_grid) * np.sin(phi_grid)
    z = diameter / 2 * np.cos(theta_grid)
    
    # Create faces and vertices using surf2patch equivalent
    if triangles:
        faces, verts = surf2patch(x, y, z, triangles=True)
        p = particle(verts, faces).clean()
    else:
        faces, verts = surf2patch(x, y, z)
        p = particle(verts, faces).clean()
    
    # Vertices and faces for curved particle boundary
    # Add midpoints
    p = p.midpoints(flat=True)
    
    # Rescale vertices to sphere surface
    # Calculate norms of each vertex
    norms = np.sqrt(np.sum(p.verts2**2, axis=1))
    
    # Rescale vertices to sphere surface
    verts2 = 0.5 * diameter * (p.verts2 / norms.reshape(-1, 1))
    
    # Make particle including midpoints
    p = particle(verts2, p.faces2, **kwargs)
    
    return p


def surf2patch(x, y, z, triangles=False):
    """
    Convert surface data to patch format (vertices and faces).
    
    Parameters:
    -----------
    x, y, z : ndarray
        Surface coordinate arrays
    triangles : bool
        If True, create triangular faces; otherwise quadrilateral faces
        
    Returns:
    --------
    faces : ndarray
        Face connectivity array
    verts : ndarray
        Vertex coordinates array
    """
    
    # Flatten coordinate arrays and stack to create vertices
    verts = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    
    # Create faces based on grid connectivity
    m, n = x.shape
    faces = []
    
    for i in range(m - 1):
        for j in range(n - 1):
            # Vertex indices for current quad
            v1 = i * n + j
            v2 = i * n + j + 1
            v3 = (i + 1) * n + j + 1
            v4 = (i + 1) * n + j
            
            if triangles:
                # Split quad into two triangles
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
            else:
                # Keep as quad
                faces.append([v1, v2, v3, v4])
    
    return np.array(faces), verts