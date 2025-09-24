import numpy as np
from typing import Union, List, Any


def tritorus(diameter, rad, *args, **kwargs):
    """
    Faces and vertices of triangulated torus.
    
    Usage:
        p = tritorus(diameter, rad, **kwargs)
        p = tritorus(diameter, rad, n, **kwargs)
        p = tritorus(diameter, rad, n, triangles=True, **kwargs)
        
    Parameters:
    -----------
    diameter : float
        Diameter of folded cylinder
    rad : float
        Radius of torus
    n : int or list of int, optional
        Number of discretization points (default: [21, 21])
        If single int, used for both directions
    triangles : bool, optional
        Use triangles rather than quadrilaterals (default: False)
    **kwargs : dict
        Additional arguments to be passed to particle constructor
        
    Returns:
    --------
    particle
        Faces and vertices of triangulated torus
    """
    
    # Extract number of discretization points
    if args and isinstance(args[0], (int, list, tuple, np.ndarray)):
        n = args[0]
        remaining_args = args[1:]
        
        # Convert single number to pair
        if isinstance(n, (int, np.integer)):
            n = [n, n]
        elif len(n) == 1:
            n = [n[0], n[0]]
    else:
        n = [21, 21]
        remaining_args = args
    
    # Extract triangles option
    triangles = kwargs.pop('triangles', False)
    
    # Check for 'triangles' string in remaining args (MATLAB style)
    if remaining_args and isinstance(remaining_args[0], str) and remaining_args[0] == 'triangles':
        triangles = True
        remaining_args = remaining_args[1:]
    
    # Grid triangulation
    phi_vals = np.linspace(0, 2 * np.pi, n[0])
    theta_vals = np.linspace(0, 2 * np.pi, n[1])
    
    verts, faces = fvgrid(phi_vals, theta_vals, triangles)
    
    # Extract angles
    phi = verts[:, 0]
    theta = verts[:, 1]
    
    # Coordinates of torus using parametric equations
    x = (0.5 * diameter + rad * np.cos(theta)) * np.cos(phi)
    y = (0.5 * diameter + rad * np.cos(theta)) * np.sin(phi)
    z = rad * np.sin(theta)
    
    # Combine coordinates
    coords = np.column_stack([x, y, z])
    
    # Make torus
    p = particle(coords, faces, **kwargs).clean()
    
    return p


def fvgrid(phi_vals, theta_vals, triangles=False):
    """
    Create grid triangulation for parametric surfaces.
    
    Parameters:
    -----------
    phi_vals : array-like
        Values for first parameter
    theta_vals : array-like  
        Values for second parameter
    triangles : bool or str
        If True or 'triangles', use triangular faces
        
    Returns:
    --------
    verts : ndarray
        Vertex parameter coordinates
    faces : ndarray
        Face connectivity array
    """
    
    # Create meshgrid
    phi_grid, theta_grid = np.meshgrid(phi_vals, theta_vals)
    
    # Flatten to create vertices (parameter space)
    verts = np.column_stack([phi_grid.flatten(), theta_grid.flatten()])
    
    # Create faces based on grid connectivity
    m, n = phi_grid.shape
    faces = []
    
    for i in range(m - 1):
        for j in range(n - 1):
            # Vertex indices for current quad
            v1 = i * n + j
            v2 = i * n + j + 1  
            v3 = (i + 1) * n + j + 1
            v4 = (i + 1) * n + j
            
            if triangles == 'triangles' or triangles is True:
                # Split quad into two triangles
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
            else:
                # Keep as quad
                faces.append([v1, v2, v3, v4])
    
    return verts, np.array(faces)