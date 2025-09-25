import numpy as np

def vecnorm(v, key=None):
    """
    VECNORM - Norm of vector array.
    
    Parameters:
    -----------
    v : array_like
        Vector array of shape (..., 3, ...)
    key : str, optional
        If 'max', returns maximum norm value
        
    Returns:
    --------
    np.ndarray or float
        Norm array with dimension 2 squeezed out, or max value if key='max'
    """
    v = np.asarray(v)
    
    # Calculate norm along axis 1 (the vector dimension)
    n = np.sqrt(np.sum(np.abs(v)**2, axis=1))
    
    # Remove singleton dimension
    n = np.squeeze(n)
    
    # Return maximum if 'max' keyword specified
    if key == 'max':
        return np.max(n)
    
    return n