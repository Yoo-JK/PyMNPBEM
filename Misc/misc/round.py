import numpy as np

def round(x, n):
    """
    ROUND - Rounds each element of X to the left of the decimal point.
    
    Parameters:
    -----------
    x : array_like
        Argument to round
    n : int
        Number of digits to the left of decimal point to round to
        
    Returns:
    --------
    np.ndarray
        Rounded argument
    """
    x = np.asarray(x)
    return np.round(x * 10**n) * 10**(-n)