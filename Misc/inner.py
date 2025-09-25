import numpy as np

def inner(nvec, a, mul=None):
    """
    INNER - Inner product between a vector and a matrix or tensor.
    
    The generalized dot product is defined with respect to the second
    dimension. If an additional mul object is provided, a -> mul * a 
    is performed prior to the inner product.
    
    Parameters:
    -----------
    nvec : array_like
        Vector for inner product
    a : array_like
        Matrix or tensor
    mul : array_like, optional
        Multiplication matrix applied to 'a' before inner product
        
    Returns:
    --------
    np.ndarray or float
        Inner product result. Returns 0 if dimensions don't match.
    """
    nvec = np.asarray(nvec)
    a = np.asarray(a)
    
    if nvec.shape[0] != a.shape[0]:
        return 0
    
    # Apply multiplication if provided
    if mul is not None:
        mul = np.asarray(mul)
        a = np.matmul(mul, a)
    
    if a.ndim == 2:
        # For 2D arrays, compute dot product along axis 1 (second dimension)
        val = np.sum(nvec * a, axis=1)
    else:
        # For higher dimensional tensors
        siz = a.shape
        
        # Replicate nvec to match tensor dimensions
        nvec_expanded = np.broadcast_to(
            nvec[:, :, np.newaxis], 
            (nvec.shape[0], nvec.shape[1], *siz[2:])
        )
        
        # Compute dot product along axis 1
        dot_result = np.sum(nvec_expanded * a, axis=1)
        
        # Reshape to remove the collapsed dimension
        val = dot_result.reshape((siz[0], *siz[2:]))
    
    return val