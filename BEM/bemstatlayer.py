"""
BEM solver for quasistatic approximation and layer structure.

Given an external excitation, BEMSTATLAYER computes the surface
charges such that the boundary conditions of Maxwell's equations
in the quasistatic approximation are fulfilled.
"""

import numpy as np
from scipy.sparse import diags as spdiag
from scipy.linalg import inv
from ..Base.bembase import BemBase
from ...utils.compstruct import CompStruct
from ...utils.compgreenstatlayer import compgreenstatlayer
from ...utils.matrix_operations import matmul


class BemStatLayer(BemBase):
    """BEM solver for quasistatic approximation and layer structure."""
    
    # Class constants
    name = 'bemsolver'
    needs = [{'sim': 'stat'}, 'layer']
    
    def __init__(self, p, *args, **kwargs):
        """
        Initialize quasistatic BEM solver for layer structure.
        
        Parameters
        ----------
        p : object
            Composite particle
        enei : float, optional
            Light wavelength in vacuum
        op : dict, optional
            Options dictionary
        **kwargs : dict
            Additional options as keyword arguments
        """
        super().__init__()
        self.p = None           # composite particle
        self.g = None           # Green function object
        self.enei = None        # light wavelength in vacuum
        
        # Private attributes
        self._mat = None        # matrix for solution of BEM equations
        
        self._init(p, *args, **kwargs)
    
    def __str__(self):
        """String representation."""
        return f"bemstatlayer: p={self.p}, g={self.g}"
    
    def display(self):
        """Command window display."""
        print("bemstatlayer:")
        print({'p': self.p, 'g': self.g})
    
    def _init(self, p, *args, **kwargs):
        """
        Initialize quasistatic BEM solver for layer structure.
        
        Parameters
        ----------
        p : object
            Composite particle
        enei : float, optional
            Light wavelength in vacuum (if first arg is numeric)
        **kwargs : dict
            Additional options
        """
        # Save particle
        self.p = p
        
        # Handle calls with and without enei
        enei = None
        varargin = list(args)
        if varargin and isinstance(varargin[0], (int, float)):
            enei = varargin.pop(0)
        
        # Convert remaining args and kwargs to single dict
        options = dict(kwargs)
        for i in range(0, len(varargin), 2):
            if i + 1 < len(varargin):
                options[varargin[i]] = varargin[i+1]
        
        # Option array - use empty dict if no options provided
        if not options:
            options = {}
        
        # Green function
        self.g = compgreenstatlayer(p, p, **options)
        
        # Initialize for given wavelength
        if enei is not None:
            self._initialize_matrices(enei)
    
    def _initialize_matrices(self, enei):
        """
        Initialize matrices for given energy/wavelength.
        
        Parameters
        ----------
        enei : float
            Light wavelength in vacuum
        """
        # Use previously computed matrices?
        if self.enei is None or self.enei != enei:
            # Inside and outside dielectric function
            eps1 = spdiag(self.p.eps1(enei))
            eps2 = spdiag(self.p.eps2(enei))
            
            # Green functions
            H1, H2 = self.g.eval(enei, 'H1', 'H2')
            
            # BEM resolvent matrix
            self._mat = -inv(eps1 @ H1 - eps2 @ H2) @ (eps1 - eps2)
            
            # Save energy
            self.enei = enei
    
    def __call__(self, enei):
        """
        Initialize BEM solver for given energy/wavelength.
        
        Parameters
        ----------
        enei : float
            Light wavelength in vacuum
            
        Returns
        -------
        self : BemStatLayer
            Self for method chaining
        """
        self._initialize_matrices(enei)
        return self
    
    def solve(self, exc):
        """
        Surface charge for given excitation (mldivide equivalent).
        
        Parameters
        ----------
        exc : CompStruct
            CompStruct with field 'phip' for external excitation
            
        Returns
        -------
        sig : CompStruct
            CompStruct with field for surface charge
        """
        # Initialize BEM solver (if needed)
        self._initialize_matrices(exc.enei)
        
        sig_values = matmul(self._mat, exc.phip)
        return CompStruct(self.p, exc.enei, sig=sig_values)
    
    def __truediv__(self, exc):
        """Operator overload for solve (\ in MATLAB)."""
        return self.solve(exc)
    
    def induced_potential(self, sig):
        """
        Induced potential for given surface charge (mtimes equivalent).
        
        Parameters
        ----------
        sig : CompStruct
            CompStruct with fields for surface charge
            
        Returns
        -------
        phi : CompStruct
            CompStruct with fields for induced potential
        """
        return self.potential(sig, 1) + self.potential(sig, 2)
    
    def __mul__(self, sig):
        """Operator overload for induced_potential (* in MATLAB)."""
        return self.induced_potential(sig)
    
    def field(self, sig, inout=2):
        """
        Electric field inside/outside of particle surface.
        
        Computed from solutions of Maxwell equations within the
        quasistatic approximation.
        
        Parameters
        ----------
        sig : CompStruct
            CompStruct object with surface charges
        inout : int, optional
            Electric field inside (inout=1) or outside (inout=2, default)
            
        Returns
        -------
        field : CompStruct
            CompStruct object with electric field
        """
        # Field from Green function
        return self.g.field(sig, inout)
    
    def potential(self, sig, inout=2):
        """
        Determine potentials and surface derivatives inside/outside of particle.
        
        Computed from solutions of Maxwell equations within the
        quasistatic approximation.
        
        Parameters
        ----------
        sig : CompStruct
            CompStruct object with surface charges
        inout : int, optional
            Potential inside (inout=1) or outside (inout=2, default)
            
        Returns
        -------
        pot : CompStruct
            CompStruct object with potentials
        """
        # Compute potential
        return self.g.potential(sig, inout)


def bemstatlayer(p, *args, **kwargs):
    """
    Factory function to create BemStatLayer instance.
    
    Parameters
    ----------
    p : object
        Composite particle
    *args : tuple
        Additional positional arguments
    **kwargs : dict
        Additional keyword arguments
        
    Returns
    -------
    BemStatLayer
        Initialized BEM solver instance
    """
    return BemStatLayer(p, *args, **kwargs)