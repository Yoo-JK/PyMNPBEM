"""
green_functions.py - Green Function Operations for H-Matrices
Converted from: hmatgreenstat.cpp, hmatgreenret.cpp, hmatgreentab1.cpp, hmatgreentab2.cpp

This module provides Green function operations:
- Static Green functions
- Retarded Green functions  
- Tabulated Green functions (G and F derivatives)
"""

import numpy as np
from typing import Union, Optional, Dict, Any, Tuple
from .acagreen import GreenStat, GreenRet, Particle
from .hlib import HMatrix, SubMatrix, Matrix, tree, hopts, timer, tic, toc

def hmat_green_static(particle_data: Dict, flag: str, options: Optional[Dict] = None) -> Tuple[list, list]:
    """
    Compute static Green function using ACA.
    
    Equivalent to hmatgreenstat.cpp functionality.
    
    Parameters:
    -----------
    particle_data : dict
        Particle data with 'pos', 'nvec', 'area' fields
    flag : str
        'G' for Green function or 'F' for surface derivative
    options : dict, optional
        Options with 'htol' and 'kmax'
        
    Returns:
    --------
    tuple
        (L_cells, R_cells) - Low-rank factor cell arrays
    """
    if options:
        if 'htol' in options:
            hopts.tol = options['htol']
        if 'kmax' in options:
            hopts.kmax = options['kmax']
    
    # Create particle object
    particle = Particle.from_dict(particle_data)
    
    # Set up Green function object
    green_func = GreenStat(particle, flag)
    
    # Compute low-rank approximation
    hmat = green_func.eval(hopts.tol)
    
    # Extract cell arrays for output
    L_cells = []
    R_cells = []
    
    if isinstance(hmat, dict):
        for (row, col), submat in hmat.items():
            if submat and hasattr(submat, 'flag') and submat.flag() == 1:  # FLAG_RK
                L_cells.append(submat.lhs.val)
                R_cells.append(submat.rhs.val)
            else:
                L_cells.append(None)
                R_cells.append(None)
    
    return L_cells, R_cells

def hmat_green_retarded(particle_data: Dict, flag: str, i: int, j: int, 
                       wavenumber: complex, options: Optional[Dict] = None) -> Tuple[list, list]:
    """
    Compute retarded Green function using ACA.
    
    Equivalent to hmatgreenret.cpp functionality.
    
    Parameters:
    -----------
    particle_data : dict
        Particle data with 'pos', 'nvec', 'area' fields
    flag : str
        'G' for Green function or 'F' for surface derivative
    i, j : int
        Starting cluster indices
    wavenumber : complex
        Wave number k = 2π/λ
    options : dict, optional
        Options with 'htol' and 'kmax'
        
    Returns:
    --------
    tuple
        (L_cells, R_cells) - Low-rank factor cell arrays
    """
    if options:
        if 'htol' in options:
            hopts.tol = options['htol']
        if 'kmax' in options:
            hopts.kmax = options['kmax']
    
    # Create particle object
    particle = Particle.from_dict(particle_data)
    
    # Set up retarded Green function object
    green_func = GreenRet(particle, flag, wavenumber)
    
    # Compute low-rank approximation for specific particle pair
    hmat = green_func.eval(i, j, hopts.tol)
    
    # Extract cell arrays for output
    L_cells = []
    R_cells = []
    
    if isinstance(hmat, dict):
        for (row, col), submat in hmat.items():
            if submat and hasattr(submat, 'flag') and submat.flag() == 1:  # FLAG_RK
                L_cells.append(submat.lhs.val)
                R_cells.append(submat.rhs.val)
            else:
                L_cells.append(None)
                R_cells.append(None)
    
    return L_cells, R_cells

def hmat_green_tabulated_g(particle_data: Dict, i: int, j: int, 
                          table_data: Dict, options: Optional[Dict] = None) -> Tuple[list, list]:
    """
    Compute tabulated Green function G using interpolation.
    
    Equivalent to hmatgreentab1.cpp functionality.
    
    Parameters:
    -----------
    particle_data : dict
        Particle data
    i, j : int
        Starting cluster indices
    table_data : dict
        Interpolation table with 'r', 'z1', 'z2', 'G', 'rmod', 'zmod' fields
    options : dict, optional
        Options with 'htol' and 'kmax'
        
    Returns:
    --------
    tuple
        (L_cells, R_cells) - Low-rank factor cell arrays
    """
    if options:
        if 'htol' in options:
            hopts.tol = options['htol']
        if 'kmax' in options:
            hopts.kmax = options['kmax']
    
    # Create particle object
    particle = Particle.from_dict(particle_data)
    
    # Determine if 2D or 3D interpolation based on z2 field
    if 'z2' in table_data and np.isscalar(table_data['z2']):
        # 2D interpolation
        layer_index = 1  # This would come from input parameters
        from .acagreen import GreenTabG2
        green_tab = GreenTabG2(particle, table_data, layer_index)
    else:
        # 3D interpolation
        from .acagreen import GreenTabG3
        green_tab = GreenTabG3(particle, table_data)
    
    # Compute Green function matrix
    hmat = green_tab.eval(i, j, hopts.tol)
    
    # Extract cell arrays
    L_cells = []
    R_cells = []
    
    if isinstance(hmat, dict):
        for (row, col), submat in hmat.items():
            if submat and hasattr(submat, 'flag') and submat.flag() == 1:  # FLAG_RK
                L_cells.append(submat.lhs.val)
                R_cells.append(submat.rhs.val)
            else:
                L_cells.append(None)
                R_cells.append(None)
    
    return L_cells, R_cells

def hmat_green_tabulated_f(particle_data: Dict, i: int, j: int, 
                          table_data: Dict, options: Optional[Dict] = None) -> Tuple[list, list]:
    """
    Compute tabulated Green function surface derivative F using interpolation.
    
    Equivalent to hmatgreentab2.cpp functionality.
    
    Parameters:
    -----------
    particle_data : dict
        Particle data
    i, j : int
        Starting cluster indices  
    table_data : dict
        Interpolation table with 'r', 'z1', 'z2', 'Fr', 'Fz', 'rmod', 'zmod' fields
    options : dict, optional
        Options with 'htol' and 'kmax'
        
    Returns:
    --------
    tuple
        (L_cells, R_cells) - Low-rank factor cell arrays
    """
    if options:
        if 'htol' in options:
            hopts.tol = options['htol']
        if 'kmax' in options:
            hopts.kmax = options['kmax']
    
    # Create particle object
    particle = Particle.from_dict(particle_data)
    
    # Determine if 2D or 3D interpolation
    if 'z2' in table_data and np.isscalar(table_data['z2']):
        # 2D interpolation for surface derivatives
        layer_index = 1  # This would come from input parameters
        from .acagreen import GreenTabF2
        green_tab = GreenTabF2(particle, table_data, layer_index)
    else:
        # 3D interpolation for surface derivatives
        from .acagreen import GreenTabF3
        green_tab = GreenTabF3(particle, table_data)
    
    # Compute surface derivative matrix
    hmat = green_tab.eval(i, j, hopts.tol)
    
    # Extract cell arrays
    L_cells = []
    R_cells = []
    
    if isinstance(hmat, dict):
        for (row, col), submat in hmat.items():
            if submat and hasattr(submat, 'flag') and submat.flag() == 1:  # FLAG_RK
                L_cells.append(submat.lhs.val)
                R_cells.append(submat.rhs.val)
            else:
                L_cells.append(None)
                R_cells.append(None)
    
    return L_cells, R_cells

class GreenFunctionOps:
    """Class wrapper for Green function operations."""
    
    @staticmethod
    def static_green(particle_data: Dict, flag: str, **options) -> Tuple[list, list]:
        """Compute static Green function."""
        return hmat_green_static(particle_data, flag, options)
    
    @staticmethod
    def retarded_green(particle_data: Dict, flag: str, i: int, j: int, 
                      wavenumber: complex, **options) -> Tuple[list, list]:
        """Compute retarded Green function."""
        return hmat_green_retarded(particle_data, flag, i, j, wavenumber, options)
    
    @staticmethod
    def tabulated_green_g(particle_data: Dict, i: int, j: int, 
                         table_data: Dict, **options) -> Tuple[list, list]:
        """Compute tabulated Green function G."""
        return hmat_green_tabulated_g(particle_data, i, j, table_data, options)
    
    @staticmethod
    def tabulated_green_f(particle_data: Dict, i: int, j: int, 
                         table_data: Dict, **options) -> Tuple[list, list]:
        """Compute tabulated Green function F (surface derivative)."""
        return hmat_green_tabulated_f(particle_data, i, j, table_data, options)