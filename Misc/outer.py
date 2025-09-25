import numpy as np

def outer(nvec, val, mul=None):
    """
    OUTER - Outer product between vector and tensor.
    
    Computes outer product where:
    a(i, 1, j, ...) = nvec(i, 1) * val(i, j, ...)
    a(i, 2, j, ...) = nvec(i, 2) * val(i, j, ...)
    a(i, 3, j, ...) = nvec(i, 3) * val(i, j, ...)
    
    If an additional matrix is provided, val -> mul * val prior to the outer product.
    
    Parameters:
    -----------
    nvec : array_like
        Vector of shape (N, 3)
    val : array_like
        Tensor of shape (N, ...)
    mul : array_like, optional
        Matrix to multiply with val before outer product
        
    Returns:
    --------
    np.ndarray or scalar
        Outer product result with shape (N, 3, ...) or 0 if dimensions don't match
    """
    nvec = np.asarray(nvec)
    val = np.asarray(val)
    
    # Apply multiplication if provided
    if mul is not None:
        from .matmul import matmul  # Import from local matmul function
        val = matmul(mul, val)
    
    # Check dimension compatibility
    if nvec.shape[0] != val.shape[0]:
        return 0
    
    # Get output shape: insert dimension 1 after first dimension
    val_shape = val.shape
    output_shape = (val_shape[0], 3) + val_shape[1:]
    
    # Initialize result array
    a = np.zeros(output_shape, dtype=val.dtype)
    
    # Compute outer product for each component
    for i in range(3):
        # Extract i-th component of nvec
        nvec_component = nvec[:, i]
        
        # Compute element-wise product with broadcasting
        if val.ndim == 1:
            # Simple vector case
            component_result = nvec_component * val
        else:
            # Tensor case - need to broadcast properly
            # Reshape nvec component to match broadcasting requirements
            broadcast_shape = [nvec.shape[0]] + [1] * (val.ndim - 1)
            nvec_reshaped = nvec_component.reshape(broadcast_shape)
            component_result = nvec_reshaped * val
        
        # Assign to result
        a[:, i, ...] = component_result
    
    # Check if result is all zeros
    if np.all(a == 0):
        return 0
    
    return a


def _resh(a, siz):
    """
    RESH - Helper function to reshape vector component.
    
    Parameters:
    -----------
    a : array_like or scalar
        Array to reshape
    siz : tuple
        Target shape
        
    Returns:
    --------
    np.ndarray
        Reshaped array
    """
    if np.isscalar(a) and a == 0:
        return np.zeros(siz)
    else:
        return np.asarray(a).reshape(siz)