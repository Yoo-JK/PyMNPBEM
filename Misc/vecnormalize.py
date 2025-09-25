import numpy as np

def vecnormalize(v, v2=None, key=''):
    """
    VECNORMALIZE - Normalize vector array.
    
    Parameters:
    -----------
    v : array_like
        Vector array of shape (..., 3, ...)
    v2 : array_like, optional
        Use another vector array for normalization (same shape as v)
    key : str, optional
        Normalization mode:
        - '': normal normalization by individual vector norms
        - 'max': normalize using maximum norm of entire array
        - 'max2': normalize each slice by its maximum norm
        
    Returns:
    --------
    np.ndarray
        Normalized vector array
    """
    v = np.asarray(v)
    
    # Use v2 for norm calculation if provided, otherwise use v
    if v2 is None:
        v2 = v
    else:
        v2 = np.asarray(v2)
    
    # Calculate norms along axis 1 (vector dimension)
    n = np.sqrt(np.sum(np.abs(v2)**2, axis=1))
    
    # Apply normalization based on key
    if key == 'max':
        # Normalize by global maximum
        v = v / np.max(n)
    elif key == 'max2':
        # Normalize each slice by its maximum norm
        max_norms = np.max(n, axis=0, keepdims=True)
        # Reshape for broadcasting
        broadcast_shape = [1] * v.ndim
        broadcast_shape[0] = v.shape[0]
        broadcast_shape[1] = v.shape[1]
        v = v / max_norms.reshape(broadcast_shape)
    else:
        # Standard normalization - divide by individual norms
        # Reshape n for broadcasting
        broadcast_shape = [1] * v.ndim
        broadcast_shape[1] = v.shape[1]
        n_broadcast = n.reshape(v.shape[0], 1, *v.shape[2:])
        
        # Avoid division by zero
        n_broadcast = np.where(n_broadcast == 0, 1, n_broadcast)
        v = v / n_broadcast
    
    return v