import numpy as np
from . import refinestat, refineret


def refine(p, ind):
    """
    Refine function to be passed to Green function initialization.
    Green function elements for neighbour cover layer elements are refined
    through polar integration.
    
    Parameters
    ----------
    p : object
        COMPARTICLE object
    ind : array_like
        Particle indices for refinement
        
    Returns
    -------
    function
        Refinement function for Green function initialization
    """
    # Symmetrize indices
    ind = np.unique(np.vstack([ind, np.fliplr(ind)]), axis=0)
    
    # Refinement function
    def fun(obj, g, f):
        return refun(obj, g, f, p, ind)
    
    return fun


def refun(obj, g, f, p, ind):
    """
    Refinement function for Green function initialization.
    
    Parameters
    ----------
    obj : object
        Green function object
    g : array_like
        Green function matrix
    f : array_like
        Surface derivative matrix
    p : object
        Particle object
    ind : array_like
        Indices for refinement
        
    Returns
    -------
    tuple
        Refined Green function and surface derivative matrices
    """
    # Select between static and retarded Green functions
    class_name = obj.__class__.__name__
    
    if class_name == 'GreenStat' or 'greenstat' in class_name.lower():
        g, f = refinestat.refine_stat(obj, g, f, p, ind)
    elif class_name == 'GreenRet' or 'greenret' in class_name.lower():
        g, f = refineret.refine_ret(obj, g, f, p, ind)
    else:
        raise ValueError(f'Unknown class name: {class_name}')
    
    return g, f