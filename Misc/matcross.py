import numpy as np

def matcross(a, b):
    """
    MATCROSS - Generalized cross product for tensors.
    
    Parameters:
    -----------
    a : array_like
        Vector of dimensions [n x 3]
    b : array_like  
        Tensor of dimensions [n x 3 x ...]
        
    Returns:
    --------
    np.ndarray
        Cross product of a and b with same dimensions as b
    """
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Get size of tensor b, set second dimension to 1 for reshaping
    siz = list(b.shape)
    siz[1] = 1
    
    def fun(i, j):
        """Helper function for cross product components"""
        # Extract b[:, j, :] and multiply with a[:, i]
        b_slice = b[:, j, ...]
        a_component = a[:, i]
        
        # Broadcast multiply and reshape
        result = b_slice * a_component[..., np.newaxis]
        return result.reshape(siz)
    
    # Compute cross product components
    # c = a × b = [a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1]
    c1 = fun(1, 2) - fun(2, 1)  # a_y * b_z - a_z * b_y
    c2 = fun(2, 0) - fun(0, 2)  # a_z * b_x - a_x * b_z  
    c3 = fun(0, 1) - fun(1, 0)  # a_x * b_y - a_y * b_x
    
    # Concatenate along second dimension
    c = np.concatenate([c1, c2, c3], axis=1)
    
    return c