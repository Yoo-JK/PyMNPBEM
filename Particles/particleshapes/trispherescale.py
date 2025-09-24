import numpy as np
from typing import Union


def trispherescale(p, scale, unit=False):
    """
    Deform surface of sphere.
    
    Parameters:
    -----------
    p : particle or comparticle object
        The particle(s) to deform
    scale : array-like
        Scale factors for deformation
    unit : bool, optional
        If True, normalize scale to maximum value (default: False)
    
    Returns:
    --------
    particle or comparticle
        Deformed particle(s)
    """
    
    # Convert scale to numpy array
    scale = np.asarray(scale)
    
    # Normalize scale if unit is True
    if unit:
        scale = scale / np.max(scale)
    
    # Check if p is a comparticle
    if hasattr(p, 'p') and hasattr(p, 'index'):  # comparticle
        # Process each particle in comparticle
        for i in range(len(p.p)):
            p.p[i] = trispherescale(p.p[i], scale[p.index[i]])
        
        # Normalize the comparticle
        p = p.norm()
        
    else:  # single particle
        # If scale length matches number of faces, interpolate to vertices
        if len(scale) == p.nfaces:
            scale = p.interp(scale)
        
        # Apply scaling to vertices
        # Reshape scale to column vector and replicate for 3D coordinates
        scale_matrix = np.tile(scale.reshape(-1, 1), (1, 3))
        p.verts = scale_matrix * p.verts
    
    return p