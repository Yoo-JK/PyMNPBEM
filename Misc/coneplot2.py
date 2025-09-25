import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation

def coneplot2(pos, vec, **kwargs):
    """
    CONEPLOT2 - Plot vectors at given positions using cones.
    
    Parameters:
    -----------
    pos : array_like
        Positions where cones are plotted
    vec : array_like
        Vectors to be plotted
    **kwargs : dict
        Additional properties:
        scale : float, optional
            Scaling factor for vectors (default: 1)
        sfun : callable, optional
            Scaling function for vectors (default: identity function)
            Example: sfun = lambda x: np.sqrt(x)
            
    Returns:
    --------
    object
        Handle to the plotted cones
    """
    pos = np.asarray(pos)
    vec = np.asarray(vec)
    
    # Check if new figure is needed
    new = len(plt.get_fignums()) == 0 or plt.gca() is None
    
    # Set default values
    scale_factor = kwargs.get('scale', 1.0)
    sfun = kwargs.get('sfun', lambda x: x)
    
    # Vector length
    vec_len = np.linalg.norm(vec, axis=1)
    
    # Scaling function
    if np.max(vec_len) > 0:
        scale = scale_factor * sfun(vec_len / np.max(vec_len))
    else:
        scale = np.ones_like(vec_len)
    
    # Cone plot
    h = _coneplot_internal(pos, vec, vec_len, scale)
    
    # Plot options
    if new:
        ax = plt.gca()
        if hasattr(ax, 'zaxis'):  # 3D plot
            ax.set_aspect('equal')
            ax.view_init(elev=40, azim=1)
        ax.axis('off')
    
    return h


def _coneplot_internal(pos, vec, vec_len, scale):
    """
    Internal cone plot function for vectors at positions.
    
    Parameters:
    -----------
    pos : np.ndarray
        Positions of cones
    vec : np.ndarray
        Vector directions
    vec_len : np.ndarray
        Vector magnitudes
    scale : np.ndarray
        Scaling factors
        
    Returns:
    --------
    object
        Patch handle
    """
    # Create cone geometry using cylinder
    theta = np.linspace(0, 2*np.pi, 21)  # 20 segments
    
    # Cone profile: tip, wide base, narrow base, shaft bottom, shaft top
    r_profile = 0.6 * np.array([0, 1, 0.5, 0.5, 0])
    z_profile = np.array([2, 0, 0, -1, -1])
    
    # Generate cone vertices
    vertices = []
    for i, (r, z) in enumerate(zip(r_profile, z_profile)):
        if r == 0:  # Tip or bottom point
            vertices.append([0, 0, z])
        else:
            for t in theta[:-1]:  # Exclude last point (same as first)
                vertices.append([r * np.cos(t), r * np.sin(t), z])
    
    base_vertices = np.array(vertices)
    nverts = len(base_vertices)
    
    # Create faces (simplified triangulation)
    faces = []
    n_theta = len(theta) - 1
    
    # Tip faces
    for i in range(n_theta):
        faces.append([0, 1 + i, 1 + (i + 1) % n_theta])
    
    # Side faces between rings
    base_idx = 1
    for ring in range(len(r_profile) - 2):
        if r_profile[ring] > 0 and r_profile[ring + 1] > 0:
            next_base = base_idx + n_theta
            for i in range(n_theta):
                i_next = (i + 1) % n_theta
                faces.extend([
                    [base_idx + i, next_base + i, base_idx + i_next],
                    [base_idx + i_next, next_base + i, next_base + i_next]
                ])
            base_idx = next_base
    
    faces = np.array(faces)
    
    # Transform vectors to spherical coordinates
    x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
    phi = np.arctan2(y, x)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    
    all_vertices = []
    all_faces = []
    color_data = []
    
    for i, (p, ph, th, s, vlen) in enumerate(zip(pos, phi, theta, scale, vec_len)):
        if s <= 0:  # Skip if no scaling
            continue
            
        # Scale vertices
        scaled_verts = base_vertices * s
        
        # Create rotation matrix
        # First rotate around y-axis by theta, then around z-axis by phi
        ry = np.array([[np.cos(-th), 0, np.sin(-th)],
                       [0, 1, 0],
                       [-np.sin(-th), 0, np.cos(-th)]])
        
        rz = np.array([[np.cos(-ph), -np.sin(-ph), 0],
                       [np.sin(-ph), np.cos(-ph), 0],
                       [0, 0, 1]])
        
        rotation_matrix = rz @ ry
        
        # Apply rotation
        rotated_verts = scaled_verts @ rotation_matrix.T
        
        # Apply translation
        translated_verts = rotated_verts + p
        
        all_vertices.extend(translated_verts)
        all_faces.extend(faces + i * nverts)
        color_data.extend([vlen] * nverts)
    
    if all_vertices:
        all_vertices = np.array(all_vertices)
        all_faces = np.array(all_faces)
        color_data = np.array(color_data)
        
        # Create 3D plot
        ax = plt.gca()
        if not hasattr(ax, 'zaxis'):
            # Create 3D axes if needed
            fig = plt.gcf()
            ax = fig.add_subplot(111, projection='3d')
        
        # Create collection
        collection = Poly3DCollection(all_vertices[all_faces], alpha=0.8)
        collection.set_array(color_data[::nverts])  # One color per cone
        collection.set_edgecolors('none')
        collection.set_facecolors(plt.cm.viridis(color_data[::nverts] / np.max(color_data)))
        
        h = ax.add_collection3d(collection)
        
        # Set lighting-like appearance
        ax.set_facecolor('white')
        
        return h
    
    return None