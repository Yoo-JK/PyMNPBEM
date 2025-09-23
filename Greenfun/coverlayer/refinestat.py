import numpy as np
from .quadpol import quadpol


def refine_stat(obj, g, f, p, ind):
    """
    Refine quasistatic Green function object.
    Green function elements for neighbour cover layer elements are refined
    through polar integration.
    
    Parameters
    ----------
    obj : object
        Green function object
    g : array_like
        Green function element for additional refinement
    f : array_like
        Surface derivative of Green function
    p : object
        COMPARTICLE object
    ind : array_like
        Particle indices for refinement
        
    Returns
    -------
    tuple
        Refined Green function and surface derivative matrices
    """
    # Index to Green function elements for refinement
    i1 = p.index(ind[:, 0])
    i2 = p.index(ind[:, 1])
    
    # Particle index
    linear_ind = np.ravel_multi_index((i1, i2), (p.n, p.n))
    _, ind_map = np.isin(linear_ind, obj.ind, assume_unique=True)
    
    # Filter valid indices
    valid_mask = ind_map != 0
    i1 = i1[valid_mask]
    i2 = i2[valid_mask]
    ind_map = ind_map[valid_mask]
    
    # Integration points and weights
    pos, weight, ind2 = quadpol(p, i2)
    ind1 = i1[ind2].reshape(-1, 1)
    
    # Measure positions with respect to centroids
    x = p.pos[ind1, 0] - pos[:, 0]
    y = p.pos[ind1, 1] - pos[:, 1]
    z = p.pos[ind1, 2] - pos[:, 2]
    
    # Distance
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Green function elements
    g[ind_map] = np.bincount(ind2, weights=weight / r, minlength=len(i2))
    
    # Derivative of Green function
    fx = -np.bincount(ind2, weights=weight * x / r**3, minlength=len(i2))
    fy = -np.bincount(ind2, weights=weight * y / r**3, minlength=len(i2))
    fz = -np.bincount(ind2, weights=weight * z / r**3, minlength=len(i2))
    
    # Save surface derivative
    if obj.deriv == 'cart':
        f[ind_map, :] = np.column_stack([fx, fy, fz])
    else:
        f[ind_map] = (fx * p.nvec[i1, 0] + 
                     fy * p.nvec[i1, 1] + 
                     fz * p.nvec[i1, 2])
    
    return g, f