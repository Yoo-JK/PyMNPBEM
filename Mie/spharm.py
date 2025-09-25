"""
PyMNPBEM - Spherical harmonics calculation
Converted from MATLAB MNPBEM spharm function
"""

import numpy as np
from scipy.special import lpmv
from typing import Union


# Global factorial table for speedup (equivalent to MATLAB persistent variable)
_factab = []


def spharm(ltab: Union[list, np.ndarray], mtab: Union[list, np.ndarray], 
           theta: Union[float, np.ndarray], phi: Union[float, np.ndarray]) -> np.ndarray:
    """
    Spherical harmonics calculation.
    
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
    y : ndarray
        Spherical harmonic values
    """
    global _factab
    
    ltab = np.asarray(ltab).flatten()
    mtab = np.asarray(mtab).flatten()
    theta = np.asarray(theta).flatten()
    phi = np.asarray(phi).flatten()
    
    # Update factorial table if needed
    max_fact_needed = np.max(ltab + np.abs(mtab)) + 1
    if max_fact_needed >= len(_factab):
        _factab = []
        for i in range(max_fact_needed + 2):
            _factab.append(np.math.factorial(i))
    
    # Initialize output array
    y = np.zeros((len(ltab), len(theta)), dtype=complex)
    
    # Loop over unique ltab values
    for l in np.unique(ltab):
        # Associated Legendre polynomials for all m values
        # Note: scipy.special.lpmv handles the normalization differently than MATLAB
        cos_theta = np.cos(theta)
        
        # Index to entries with degree l
        l_indices = np.where(ltab == l)[0]
        # Filter for valid entries where abs(m) <= l
        valid_indices = l_indices[np.abs(mtab[l_indices]) <= l]
        
        for i in valid_indices:
            m = mtab[i]
            abs_m = abs(m)
            
            # Prefactor for spherical harmonics
            c = np.sqrt((2 * l + 1) / (4 * np.pi) * 
                       _factab[l - abs_m] / _factab[l + abs_m])
            
            # Associated Legendre polynomial
            # scipy's lpmv(m, l, x) corresponds to MATLAB's legendre(l, x)(m+1, :)
            plm = lpmv(abs_m, l, cos_theta)
            
            # Spherical harmonics
            y[i, :] = c * plm * np.exp(1j * abs_m * phi)
            
            # Correct for negative orders
            if m < 0:
                y[i, :] = ((-1) ** m) * np.conj(y[i, :])
    
    return y\