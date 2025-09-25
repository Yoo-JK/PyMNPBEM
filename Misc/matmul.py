import numpy as np

def matmul(a, x):
    """
    MATMUL - Generalized matrix multiplication for tensors.
    
    The matrix multiplication is performed along the last dimension of a and
    the first dimension of x.
    
    Parameters:
    -----------
    a : array_like
        Matrix or tensor for multiplication
    x : array_like
        Matrix or tensor to be multiplied
        
    Returns:
    --------
    np.ndarray or scalar
        Result of generalized matrix multiplication
    """
    a = np.asarray(a)
    x = np.asarray(x)
    
    # Handle scalar cases
    if a.size == 1:
        # a is scalar
        if a == 0:
            return 0
        else:
            return a * x
    elif x.size == 1 and x == 0:
        # x is zero
        return 0
    else:
        # a is matrix/tensor
        
        # Get sizes of matrices
        sizx = x.shape
        siza = a.shape
        
        # Check if we need diagonal multiplication
        if len(siza) == 2 and siza[-1] != sizx[0]:
            # Use diagonal of a for element-wise multiplication
            diag_a = np.diag(a)
            x_reshaped = x.reshape(sizx[0], -1)
            result = diag_a[:, np.newaxis] * x_reshaped
            y = result.reshape(sizx)
        else:
            # Standard matrix multiplication with reshaping
            
            # Calculate output size
            siz = siza[:-1] + sizx[1:]
            
            # Reshape for matrix multiplication
            a_reshaped = a.reshape(-1, siza[-1])
            x_reshaped = x.reshape(sizx[0], -1)
            
            # Perform multiplication
            result = a_reshaped @ x_reshaped
            
            # Reshape to final dimensions
            y = result.reshape(siz)
    
    return y