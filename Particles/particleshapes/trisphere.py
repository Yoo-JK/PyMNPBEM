import numpy as np
import scipy.io
from typing import Any, List, Optional
import os


def trisphere(n, *args, **kwargs):
    """
    Load points on sphere from file and perform triangulation.
    
    Given a set of stored points on the surface of a sphere, 
    loads the points and performs a triangulation.
    Sets of points which minimize the potential energy are from
    http://www.maths.unsw.edu.au/school/articles/me100.html.
    
    Parameters:
    -----------
    n : int
        Number of points for sphere triangulation
    diameter : float, optional
        Diameter of sphere (default: 1)
    **kwargs : dict
        Additional parameters to be passed to particle constructor
        
    Returns:
    --------
    particle
        Triangulated faces and vertices of sphere
    """
    
    # Extract diameter
    if args and isinstance(args[0], (int, float)):
        diameter = args[0]
        remaining_args = args[1:]
    else:
        diameter = 1
        remaining_args = args
    
    # Load data from file
    # Saved vertex point numbers available in trisphere.mat
    nsav = [32, 60, 144, 169, 225, 256, 289, 324, 361, 400, 441, 484, 
            529, 576, 625, 676, 729, 784, 841, 900, 961, 1024, 1225, 1444]
    
    # Find number closest to saved number
    diff = [abs(nsav_i - n) for nsav_i in nsav]
    ind = diff.index(min(diff))
    closest_n = nsav[ind]
    
    # Input variable name
    sphere_name = f'sphere{closest_n}'
    
    if n != closest_n:
        print(f'trisphere: loading {sphere_name} from trisphere.mat')
    
    # Load data from file
    try:
        mat_data = scipy.io.loadmat('trisphere.mat')
        sphere = mat_data[sphere_name]
        
        # Extract sphere data (assuming it's a struct with x, y, z fields)
        if sphere.dtype.names:  # struct type
            sphere_struct = sphere[0, 0]
            x = np.array(sphere_struct['x'], dtype=float).flatten()
            y = np.array(sphere_struct['y'], dtype=float).flatten()
            z = np.array(sphere_struct['z'], dtype=float).flatten()
        else:
            # If it's directly an array, assume it's [x, y, z] coordinates
            coords = np.array(sphere, dtype=float)
            if coords.shape[1] == 3:
                x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            else:
                raise ValueError("Unexpected sphere data format")
                
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not load {sphere_name} from trisphere.mat")
        print(f"Error: {e}")
        # Fallback: generate uniform sphere points
        print("Generating uniform sphere points as fallback...")
        x, y, z = _generate_uniform_sphere_points(closest_n)
    
    # Make sphere
    # Vertices
    verts = np.column_stack([x, y, z])
    
    # Face list using spherical triangulation
    faces = sphtriangulate(verts)
    
    # Rescale sphere
    verts = 0.5 * verts * diameter
    
    # Make particle
    p = particle(verts, faces, norm=False, **kwargs)
    
    # Vertices and faces for curved particle boundary
    # Add midpoints
    p = p.midpoints(flat=True)
    
    # Rescale vertices to sphere surface
    norms = np.sqrt(np.sum(p.verts2**2, axis=1))
    verts2 = 0.5 * diameter * (p.verts2 / norms.reshape(-1, 1))
    
    # Make particle including midpoints
    p = particle(verts2, p.faces2, **kwargs)
    
    return p


def _generate_uniform_sphere_points(n):
    """
    Generate approximately uniform points on unit sphere as fallback.
    
    Parameters:
    -----------
    n : int
        Approximate number of points desired
        
    Returns:
    --------
    tuple
        (x, y, z) coordinates of points on unit sphere
    """
    
    # Use golden spiral method for uniform distribution
    indices = np.arange(0, n, dtype=float) + 0.5
    
    # Golden angle in radians
    phi = np.arccos(1 - 2 * indices / n)  # latitude
    theta = np.pi * (1 + 5**0.5) * indices  # longitude
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    return x, y, z


def sphtriangulate(verts):
    """
    Perform spherical triangulation of points on sphere.
    
    Parameters:
    -----------
    verts : ndarray
        Vertices on unit sphere (n x 3)
        
    Returns:
    --------
    ndarray
        Triangle faces array (m x 3)
    """
    
    try:
        from scipy.spatial import SphericalVoronoi, geometric_slerp
        
        # Create spherical Voronoi diagram
        sv = SphericalVoronoi(verts, radius=1, center=np.array([0, 0, 0]))
        
        # Extract triangular faces from vertices
        # Each vertex corresponds to a Voronoi region
        faces = []
        
        # Simple Delaunay triangulation on sphere
        # This is a simplified implementation
        n_verts = len(verts)
        
        # For small number of points, use brute force approach
        if n_verts < 1000:
            from scipy.spatial import ConvexHull
            
            # Project to plane for triangulation, then map back
            # Use stereographic projection
            # Avoid points too close to south pole
            south_pole_threshold = -0.99
            
            # Find point furthest from south pole for projection center
            z_coords = verts[:, 2]
            if np.min(z_coords) > south_pole_threshold:
                projection_center = np.array([0, 0, -1])
            else:
                projection_center = np.array([0, 0, 1])
            
            # Stereographic projection
            projected = []
            valid_indices = []
            
            for i, v in enumerate(verts):
                # Skip if too close to projection center
                if np.dot(v, projection_center) > 0.99:
                    continue
                    
                denom = 1 - np.dot(v, projection_center)
                if abs(denom) > 1e-10:
                    proj = (v - projection_center) / denom
                    projected.append([proj[0], proj[1]])
                    valid_indices.append(i)
            
            if len(projected) >= 3:
                projected = np.array(projected)
                hull = ConvexHull(projected)
                
                # Map triangle indices back to original vertices
                faces = [[valid_indices[i] for i in tri] for tri in hull.simplices]
        
        return np.array(faces)
        
    except ImportError:
        print("Warning: scipy.spatial not available for spherical triangulation")
        print("Using simple triangulation fallback...")
        
        # Simple fallback: create faces based on ordering
        n = len(verts)
        faces = []
        
        # This is a very basic triangulation - not optimal
        for i in range(n - 2):
            faces.append([0, i + 1, i + 2])
        
        return np.array(faces)
    
    except Exception as e:
        print(f"Warning: Spherical triangulation failed: {e}")
        print("Using simple fallback triangulation...")
        
        # Fallback triangulation
        n = len(verts)
        faces = []
        for i in range(n - 2):
            faces.append([0, i + 1, i + 2])
            
        return np.array(faces)