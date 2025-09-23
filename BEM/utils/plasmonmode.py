"""
Compute plasmon eigenmodes for discretized surface.

This utility function computes plasmon eigenmodes using the BEM solver
with eigenmode expansion (bemstateig).
"""

import numpy as np
from ..BEM.bemstateig import bemstateig


def plasmonmode(p, nev=20, **kwargs):
    """
    Compute plasmon eigenmodes for discretized surface.
    
    Parameters
    ----------
    p : object
        Compound of discretized particles
    nev : int, optional
        Number of eigenmodes (default: 20)
    **kwargs : dict
        Additional arguments to be passed to bemstateig
        
    Returns
    -------
    ene : ndarray
        Eigenenergies (sorted by real part)
    ur : ndarray
        Right eigenvectors (corresponding to sorted eigenvalues)
    ul : ndarray
        Left eigenvectors (corresponding to sorted eigenvalues)
        
    Examples
    --------
    >>> # Compute 30 plasmon modes for particle p
    >>> ene, ur, ul = plasmonmode(p, nev=30)
    >>> 
    >>> # Compute modes with specific options
    >>> ene, ur, ul = plasmonmode(p, nev=50, htol=1e-4)
    """
    # Compute plasmon modes using bemstateig
    bem = bemstateig(p, nev=nev, **kwargs)
    
    # Get eigenenergies and eigenmodes
    ene = np.diag(bem.ene)
    
    # Sort by real part of eigenvalues
    ind = np.argsort(np.real(ene))
    ene = ene[ind]
    ur = bem.ur[:, ind]
    ul = bem.ul[ind, :]
    
    return ene, ur, ul