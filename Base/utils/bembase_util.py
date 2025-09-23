"""
MNPBEM Factory Functions - Python Implementation

Factory functions for creating MNPBEM objects using option structures.
These functions correspond to the original MATLAB files:
- bemsolver.m
- dipole.m  
- electronbeam.m
- greenfunction.m
- planewave.m
- spectrum.m

Translated from MATLAB MNPBEM toolbox.
"""

from typing import Any, Dict, Union, Optional
from ..base.bembase import BemBase


def bemsolver(p, *args, **kwargs):
    """
    Select appropriate BEM solver using options.
    
    Parameters:
    -----------
    p : object
        Compound of particles (see comparticle)
    *args : tuple
        Variable arguments that may include:
        - enei : light wavelength in vacuum (optional)
        - op : options dictionary
    **kwargs : dict
        Additional option fields as PropertyName, PropertyValue pairs
        
    Returns:
    --------
    bem : object
        BEM solver instance
        
    Usage:
        bem = bemsolver(p, op, **kwargs)
        bem = bemsolver(p, enei, op, **kwargs)
    """
    # Find BEM solver (deal with different call sequences)
    if args and isinstance(args[0], dict):
        # First argument is options dict
        op = args[0]
        remaining_args = args[1:]
    elif len(args) >= 2 and isinstance(args[1], dict):
        # Second argument is options dict (first might be enei)
        op = args[1]
        remaining_args = args[2:]
    else:
        # No options dict found, create empty one
        op = {}
        remaining_args = args
    
    # Add kwargs to options
    op.update(kwargs)
    
    # Find appropriate class
    class_type = BemBase.find('bemsolver', op)
    
    # Check for suitable class
    if class_type is None:
        raise ValueError('no class for options structure found')
    
    # Initialize BEM solver
    return class_type(p, *args)


def dipole(*args, **kwargs):
    """
    Initialize dipole excitation.
    
    Parameters:
    -----------
    *args : tuple
        Variable arguments including:
        - pt : compound of points (or compoint) for dipole positions
        - dip : directions of dipole moments
        - op : options dictionary
    **kwargs : dict
        Additional option fields as PropertyName, PropertyValue pairs
        
    Returns:
    --------
    exc : object
        Dipole excitation instance
        
    Usage:
        exc = dipole(pt, dip, **kwargs)
        exc = dipole(pt, dip, 'full', op, **kwargs)
    """
    # Find option structure
    op_index = None
    for i, arg in enumerate(args):
        if isinstance(arg, dict):
            op_index = i
            break
    
    if op_index is not None:
        op = args[op_index].copy()
        op.update(kwargs)
    else:
        op = kwargs.copy()
    
    # Find dipole excitation class
    class_type = BemBase.find('dipole', op)
    
    # Check for suitable class
    if class_type is None:
        raise ValueError('no class for options structure found')
    
    # Initialize dipole excitation
    return class_type(*args, **kwargs)


def electronbeam(*args, **kwargs):
    """
    Initialize electron excitation for EELS simulation.
    
    Parameters:
    -----------
    *args : tuple
        Variable arguments including:
        - p : particle object for EELS measurement
        - impact : impact parameter of electron beam
        - width : width of electron beam for potential smearing
        - vel : electron velocity in units of speed of light
        - op : option structure
    **kwargs : dict
        Additional option fields as PropertyName, PropertyValue pairs
        
    Returns:
    --------
    exc : object
        Electron beam excitation instance
        
    Usage:
        exc = electronbeam(p, impact, width, vel, op, **kwargs)
    """
    # Find option structure
    op_index = None
    for i, arg in enumerate(args):
        if isinstance(arg, dict):
            op_index = i
            break
    
    if op_index is not None:
        op = args[op_index].copy()
        op.update(kwargs)
    else:
        op = kwargs.copy()
    
    # Find electron beam excitation class
    class_type = BemBase.find('eels', op)
    
    # Check for suitable class
    if class_type is None:
        raise ValueError('no class for options structure found')
    
    # Initialize electron beam excitation
    return class_type(*args, **kwargs)


def greenfunction(*args, **kwargs):
    """
    Initialize Green function.
    
    Parameters:
    -----------
    *args : tuple
        Variable arguments including:
        - p1 : First set of points
        - p2 : Second set of points (comparticle)
        - op : options dictionary
    **kwargs : dict
        Additional option fields as PropertyName, PropertyValue pairs
        
    Returns:
    --------
    g : object
        Green function instance
        
    Usage:
        g = greenfunction(p1, p2, **kwargs)
        g = greenfunction(p1, p2, op, **kwargs)
    """
    # Find option structure
    op_index = None
    for i, arg in enumerate(args):
        if isinstance(arg, dict):
            op_index = i
            break
    
    if op_index is not None:
        op = args[op_index].copy()
        op.update(kwargs)
    else:
        op = kwargs.copy()
    
    # Find Green function class
    class_type = BemBase.find('greenfunction', op)
    
    # Check for suitable class
    if class_type is None:
        raise ValueError('no class for options structure found')
    
    # Initialize Green function
    return class_type(*args, **kwargs)


def planewave(*args, **kwargs):
    """
    Initialize plane wave excitation.
    
    Parameters:
    -----------
    *args : tuple
        Variable arguments including:
        - pol : light polarization
        - dir : propagation direction (optional)
        - op : options dictionary
    **kwargs : dict
        Additional option fields as PropertyName, PropertyValue pairs
        
    Returns:
    --------
    exc : object
        Plane wave excitation instance
        
    Usage:
        exc = planewave(pol, op, **kwargs)
        exc = planewave(pol, dir, op, **kwargs)
    """
    # Find option structure
    op_index = None
    for i, arg in enumerate(args):
        if isinstance(arg, dict):
            op_index = i
            break
    
    if op_index is not None:
        op = args[op_index].copy()
        op.update(kwargs)
    else:
        op = kwargs.copy()
    
    # Find plane wave excitation class
    class_type = BemBase.find('planewave', op)
    
    # Check for suitable class
    if class_type is None:
        raise ValueError('no class for options structure found')
    
    # Initialize plane wave excitation
    return class_type(*args, **kwargs)


def spectrum(*args, **kwargs):
    """
    Initialize spectrum.
    
    Parameters:
    -----------
    *args : tuple
        Variable arguments including:
        - pinfty : unit sphere at infinity (optional)
        - dir : light propagation direction (optional)
        - op : options dictionary
    **kwargs : dict
        Additional option fields as PropertyName, PropertyValue pairs
        Special properties:
        - 'medium' : medium for computation of spectrum
        
    Returns:
    --------
    spec : object
        Spectrum instance
        
    Usage:
        spec = spectrum(pinfty, op, **kwargs)
        spec = spectrum(dir, op, **kwargs)
        spec = spectrum(op, **kwargs)
    """
    # Find option structure
    op_index = None
    for i, arg in enumerate(args):
        if isinstance(arg, dict):
            op_index = i
            break
    
    if op_index is not None:
        op = args[op_index].copy()
        op.update(kwargs)
    else:
        op = kwargs.copy()
    
    # Find spectrum class
    class_type = BemBase.find('spectrum', op)
    
    # Check for suitable class
    if class_type is None:
        raise ValueError('no class for options structure found')
    
    # Initialize spectrum
    return class_type(*args, **kwargs)


# Convenience imports for easier access
__all__ = [
    'bemsolver',
    'dipole', 
    'electronbeam',
    'greenfunction',
    'planewave',
    'spectrum'
]