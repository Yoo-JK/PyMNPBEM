"""
PyMNPBEM - Mie theory solver factory function
Converted from MATLAB MNPBEM miesolver function
"""

from typing import Any, Dict, Union, Optional
from ..core.bembase import BEMBase
from .miestat import MieStat


def miesolver(*args, **kwargs) -> BEMBase:
    """
    Initialize solver for Mie theory.
    
    Parameters:
    -----------
    epsin : callable
        Dielectric functions inside of sphere
    epsout : callable  
        Dielectric functions outside of sphere
    diameter : float
        Sphere diameter
    op : dict, optional
        Options dictionary
    **kwargs : dict
        Additional properties. Either op or kwargs can contain
        'lmax' that determines the maximum number for spherical harmonic degrees
        
    Returns:
    --------
    mie : BEMBase
        Initialized Mie solver object
        
    Usage:
    ------
    >>> mie = miesolver(epsin, epsout, diameter, op, **kwargs)
    """
    
    # Convert args to list for easier manipulation
    args_list = list(args)
    
    # Find option structure (dict) in arguments
    op_dict = {}
    struct_index = None
    
    for i, arg in enumerate(args_list):
        if isinstance(arg, dict):
            op_dict = arg
            struct_index = i
            break
    
    # If no dict found in args, create empty one
    if struct_index is None:
        struct_index = len(args_list)
        op_dict = {}
    
    # Merge kwargs into op_dict
    op_dict.update(kwargs)
    
    # Find Mie solver class based on options
    solver_class = _find_mie_solver(op_dict)
    
    # Check for suitable class
    if solver_class is None:
        raise ValueError("No suitable Mie solver class found for given options")
    
    # Extract positional arguments (epsin, epsout, diameter)
    pos_args = args_list[:min(3, struct_index)]
    
    # Initialize BEM solver with all arguments and options
    mie = solver_class(*pos_args, **op_dict)
    
    return mie


def _find_mie_solver(options: Dict[str, Any]) -> Optional[type]:
    """
    Find appropriate Mie solver class based on options.
    
    Parameters:
    -----------
    options : dict
        Options dictionary containing solver preferences
        
    Returns:
    --------
    solver_class : type or None
        Appropriate solver class or None if not found
    """
    
    # Check for simulation type in options
    sim_type = options.get('sim', 'stat')  # Default to static
    
    # Available Mie solvers mapping
    mie_solvers = {
        'stat': MieStat,
        'static': MieStat,
        'quasistatic': MieStat,
        # Add other solver types here when implemented:
        # 'ret': MieRet,
        # 'retarded': MieRet,
        # 'full': MieFull,
    }
    
    # Try to find solver based on simulation type
    solver_class = mie_solvers.get(sim_type.lower())
    
    if solver_class is not None:
        return solver_class
    
    # If no specific sim type, try to determine from other options
    if 'lmax' in options:
        # If lmax is specified, assume quasistatic
        return MieStat
    
    # Default fallback
    return MieStat


# Alternative factory functions for specific solver types
def miestat(*args, **kwargs) -> MieStat:
    """
    Create a quasistatic Mie solver.
    
    Parameters:
    -----------
    Same as miesolver()
    
    Returns:
    --------
    mie : MieStat
        Quasistatic Mie solver object
    """
    kwargs.setdefault('sim', 'stat')
    return miesolver(*args, **kwargs)


def mie_quasistatic(*args, **kwargs) -> MieStat:
    """
    Create a quasistatic Mie solver (alias for miestat).
    
    Parameters:
    -----------
    Same as miesolver()
    
    Returns:
    --------
    mie : MieStat
        Quasistatic Mie solver object
    """
    return miestat(*args, **kwargs)


# Example usage and validation
def _validate_miesolver():
    """
    Validation function to test miesolver functionality.
    """
    try:
        # Example dielectric functions
        def epsin(enei):
            return 2.0 + 0.1j  # Simple constant dielectric function
            
        def epsout(enei):
            return 1.0  # Vacuum
        
        diameter = 10.0  # nm
        
        # Test different initialization methods
        mie1 = miesolver(epsin, epsout, diameter)
        mie2 = miesolver(epsin, epsout, diameter, {'lmax': 5})
        mie3 = miesolver(epsin, epsout, diameter, lmax=8, sim='stat')
        mie4 = miestat(epsin, epsout, diameter, lmax=10)
        
        print("All miesolver tests passed!")
        return True
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return False


if __name__ == "__main__":
    _validate_miesolver()