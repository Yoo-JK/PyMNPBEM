"""
PyMNPBEM - Table of spherical harmonic degrees and orders
Converted from MATLAB MNPBEM sphtable function
"""

import numpy as np
from typing import Tuple, Union


def sphtable(lmax: int, key: str = 'z') -> Tuple[np.ndarray, np.ndarray]:
    """
    Table of spherical harmonic degrees and orders.
    
    Parameters:
    -----------
    lmax : int
        Maximum of spherical harmonic degrees
    key : str, optional
        Keep only mtab = [-1, 0, 1] if set to 'z' (default), 
        otherwise use full range [-l, l]
        
    Returns:
    --------
    ltab : ndarray
        Table of spherical harmonic degrees
    mtab : ndarray
        Table of spherical harmonic orders
    """
    ltab = []
    mtab = []
    
    for l in range(1, lmax + 1):
        if key == 'z':
            m = np.array([-1, 0, 1])
        else:
            m = np.arange(-l, l + 1)
        
        ltab.extend([l] * len(m))
        mtab.extend(m)
    
    return np.array(ltab), np.array(mtab)