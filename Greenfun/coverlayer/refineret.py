import numpy as np
from .quadpol import quadpol
import math


def refine_ret(obj, g, f, p, ind):
    """
    Refine retarded Green function object.
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
    
    # Difference vector between face centroid and integration points
    x = p.pos[ind1, 0] - pos[:, 0]
    y = p.pos[ind1, 1] - pos[:, 1]
    z = p.pos[ind1, 2] - pos[:, 2]
    
    # Distance
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Difference vector between face centroids
    vec0 = -(p.pos[ind1, :] - p.pos[i2[ind2], :])
    
    # Distance
    r0 = np.sqrt(np.sum(vec0**2, axis=1))
    
    # Integration function
    def quad(f_vals):
        return np.bincount(ind2, weights=weight * f_vals, minlength=len(i2))
    
    # Green function
    for ord in range(obj.order + 1):
        g[ind_map, ord] = quad((r - r0)**ord / (r * math.factorial(ord)))
    
    # Surface derivative of Green function
    if obj.deriv == 'norm':
        # Inner product
        inner_prod = (x * p.nvec[ind1, 0] + 
                     y * p.nvec[ind1, 1] + 
                     z * p.nvec[ind1, 2])
        
        # Lowest order
        f[ind_map, 0] = -quad(inner_prod / r**3)
        
        # Loop over orders
        for ord in range(1, obj.order + 1):
            term1 = (r - r0)**ord / (r**3 * math.factorial(ord))
            term2 = (r - r0)**(ord-1) / (r**2 * math.factorial(ord-1))
            f[ind_map, ord] = quad(inner_prod * (term1 + term2))
    
    elif obj.deriv == 'cart':
        # Vector integration function
        def fun(f_vals):
            return np.column_stack([
                quad(x * f_vals),
                quad(y * f_vals),
                quad(z * f_vals)
            ])
        
        # Lowest order
        f[ind_map, :, 0] = -fun(1 / r**3)
        
        # Loop over orders
        for ord in range(1, obj.order + 1):
            term1 = -(r - r0)**ord / (r**3 * math.factorial(ord))
            term2 = (r - r0)**(ord-1) / (r**2 * math.factorial(ord-1))
            f[ind_map, :, ord] = fun(term1 + term2)
    
    return g, f