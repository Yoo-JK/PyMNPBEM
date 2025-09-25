"""
PyMNPBEM - Vector spherical harmonics
Converted from MATLAB MNPBEM vecspharm function
"""

import numpy as np
from typing import Tuple, Union
from .spharm import spharm


def vecspharm(ltab: Union[list, np.ndarray], mtab: Union[list, np.ndarray], 
              theta: Union[float, np.ndarray], phi: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vector spherical harmonics.
    
    Parameters:
    -----------
    ltab : array_like
        Table of spherical harmonic degrees
    mtab : array_like
        Table of spherical harmonic orders
    theta : float or array_like
        Polar angle
    phi : float or array_like
        Azimuthal angle
        
    Returns:
    --------
    x : ndarray
        Vector spherical harmonic [Jackson eq. (9.119)]
    y : ndarray
        Spherical harmonics
    """
    l = np.asarray(ltab).flatten()
    m = np.asarray(mtab).flatten()
    
    # Spherical harmonics
    y = spharm(l, m, theta, phi)
    yp = spharm(l, m + 1, theta, phi)
    ym = spharm(l, m - 1, theta, phi)
    
    # Dimension of theta and phi
    phi_flat = np.asarray(phi).flatten()
    dim = (1, len(phi_flat))
    
    # Normalization constant
    norm = 1 / np.sqrt(l * (l + 1))
    
    # Action of angular momentum operator on spherical harmonic
    # [Jackson eq. (9.104)]
    lpy = np.tile(norm * np.sqrt((l - m) * (l + m + 1)), dim) * yp
    lmy = np.tile(norm * np.sqrt((l + m) * (l - m + 1)), dim) * ym
    lzy = np.tile(norm * m, dim) * y
    
    # Vector spherical harmonics
    x = (outer(lpy, np.array([1, -1j, 0])) / 2 +
         outer(lmy, np.array([1,  1j, 0])) / 2 +
         outer(lzy, np.array([0,   0, 1])))
    
    return x, y


def outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Outer product of matrix a and vector b.
    
    Parameters:
    -----------
    a : ndarray
        Input matrix
    b : ndarray
        Input vector (length 3)
        
    Returns:
    --------
    c : ndarray
        3D array result of outer product
    """
    return np.stack([b[0] * a, b[1] * a, b[2] * a], axis=2)