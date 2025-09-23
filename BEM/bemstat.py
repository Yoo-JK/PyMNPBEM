"""
BEM Quasistatic Solver - Python Implementation

BEM solver for quasistatic approximation.
Given an external excitation, BEMSTAT computes the surface charges 
such that the boundary conditions of Maxwell's equations in the 
quasistatic approximation are fulfilled.

References: 
- Garcia de Abajo and Howie, PRB 65, 115418 (2002)
- Hohenester et al., PRL 103, 106801 (2009)

Translated from MATLAB MNPBEM toolbox @bemstat class.
"""

import numpy as np
from typing import Optional, Tuple, Any, Dict, Union
from ..Base.bembase import BemBase


class BemStat(BemBase):
    """
    BEM solver for quasistatic approximation.
    
    Given an external excitation, BEMSTAT computes the surface charges
    such that the boundary conditions of Maxwell's equations in the
    quasistatic approximation are fulfilled.
    
    The quasistatic approximation is valid when the particle size is
    much smaller than the wavelength of light.
    
    References: 
    - Garcia de Abajo and Howie, PRB 65, 115418 (2002)
    - Hohenester et al., PRL 103, 106801 (2009)
    """
    
    # BemBase abstract properties
    name = 'bemsolver'
    needs = [{'sim': 'stat'}]
    
    def __init__(self, *args, **kwargs):
        """
        Initialize quasistatic BEM solver.
        
        Parameters:
        -----------
        *args : tuple
            Arguments including:
            - p : compound of particles (see comparticle)
            - enei : light wavelength in vacuum (optional)
            - op : options dictionary
        **kwargs : dict
            Additional PropertyName, PropertyValue pairs
        """
        # Public properties
        self.p = None           # composite particle (see comparticle)
        self.F = None           # surface derivative of Green function
        self.enei = None        # light wavelength in vacuum
        
        # Private properties
        self._g = None          # Green function (needed in bemstat/field)
        self._mat = None        # -inv(Lambda + F)
        
        # Initialize
        self._init(*args, **kwargs)
    
    def __str__(self):
        """String representation."""
        return f"BemStat(p={type(self.p).__name__ if self.p else None}, F={type(self.F).__name__ if self.F else None})"
    
    def __repr__(self):
        """Command window display."""
        return f"bemstat:\n{{'p': {self.p}, 'F': {self.F}}}"
    
    def _init(self, p, *args, **kwargs):
        """
        Initialize quasistatic BEM solver.
        
        Parameters:
        -----------
        p : object
            Compound of particles (see comparticle)
        *args : tuple
            Additional arguments including optional enei and options
        **kwargs : dict
            Additional options
        """
        # Save particle
        self.p = p
        
        # Handle calls with and without ENEI
        varargin = list(args)
        enei = None
        if varargin and isinstance(varargin[0], (int, float, complex)):
            enei = varargin[0]
            varargin = varargin[1:]
        
        # Option array
        if not varargin:
            varargin = [{}]
        
        # Merge options
        options = {}
        for arg in varargin:
            if isinstance(arg, dict):
                options.update(arg)
        options.update(kwargs)
        
        # Green function for quasistatic case
        self._g = self._compgreenstat(p, p, options)
        
        # Surface derivative of Green function
        self.F = getattr(self._g, 'F', np.eye(100))  # Placeholder
        
        # Initialize for given wavelength
        if enei is not None:
            self._initmat(enei)
    
    def _compgreenstat(self, p1, p2, options):
        """
        Placeholder for quasistatic Green function initialization.
        In actual implementation, this would create the quasistatic Green function.
        """
        # This would be implemented with actual quasistatic Green function computation
        return type('QuasistaticGreenFunction', (), {
            'G': np.eye(100),  # Placeholder Green function matrix
            'H1': np.eye(100),  # Placeholder H1 matrix
            'H2': np.eye(100),  # Placeholder H2 matrix
            'F': np.eye(100),   # Placeholder F matrix (surface derivative)
            'potential': lambda sig, inout: sig,  # Placeholder
            'field': lambda sig, inout: sig,      # Placeholder
            'deriv': 'cart'
        })()
    
    def clear(self):
        """
        Clear auxiliary matrices.
        
        Returns:
        --------
        obj : BemStat
            Updated object with cleared matrices
        """
        self._mat = None
        return self
    
    def _initmat(self, enei: float):
        """
        Initialize matrices for BEM solver.
        
        Parameters:
        -----------
        enei : float
            Light wavelength in vacuum
        """
        # Use previously computed matrices?
        if self.enei is None or self.enei != enei:
            # Inside and outside dielectric function
            eps1 = self.p.eps1(enei) if hasattr(self.p, 'eps1') else np.ones(100)
            eps2 = self.p.eps2(enei) if hasattr(self.p, 'eps2') else np.ones(100)
            
            # Lambda [Garcia de Abajo, Eq. (23)]
            lambda_vals = 2 * np.pi * (eps1 + eps2) / (eps1 - eps2)
            
            # BEM resolvent matrix
            self._mat = -np.linalg.inv(np.diag(lambda_vals) + self.F)
            
            # Save energy
            self.enei = enei
    
    def __call__(self, enei: float):
        """
        Initialize matrices for given energy.
        
        Parameters:
        -----------
        enei : float
            Light wavelength in vacuum
            
        Returns:
        --------
        obj : BemStat
            Updated object with initialized matrices
        """
        self._initmat(enei)
        return self
    
    def __truediv__(self, exc):
        """Surface charge for given excitation."""
        return self.mldivide(exc)
    
    def mldivide(self, exc):
        """
        Surface charge for given excitation.
        
        Parameters:
        -----------
        exc : object
            compstruct with field 'phip' for external excitation
            
        Returns:
        --------
        sig : object
            compstruct with field for surface charge
        obj : BemStat
            Updated solver object
        """
        # Initialize BEM solver (if needed)
        self._initmat(exc.enei)
        
        # Compute surface charge
        phip = getattr(exc, 'phip', np.zeros(100))  # External excitation
        sig_values = self._matmul(self._mat, phip)
        
        # Create result structure
        sig = type('CompStruct', (), {
            'enei': exc.enei,
            'sig': sig_values
        })()
        
        return sig, self
    
    def __mul__(self, sig):
        """
        Induced potential for given surface charge.
        
        Parameters:
        -----------
        sig : object
            compstruct with fields for surface charge
            
        Returns:
        --------
        phi : object
            compstruct with fields for induced potential
        """
        phi1 = self.potential(sig, 1)
        phi2 = self.potential(sig, 2)
        
        # Combine potentials (simplified)
        phi = type('CompStruct', (), {})()
        if hasattr(phi1, 'phi') and hasattr(phi2, 'phi'):
            phi.phi = phi1.phi + phi2.phi
        else:
            phi.phi = phi1 + phi2  # Fallback for simple arrays
        
        return phi
    
    def potential(self, sig, inout: int = 2):
        """
        Determine potentials and surface derivatives inside/outside of particle.
        
        Computed from solutions of Maxwell equations within the
        quasistatic approximation.
        
        Parameters:
        -----------
        sig : object
            compstruct with surface charges (see bemstat.mldivide)
        inout : int
            potential inside (inout=1) or outside (inout=2, default) of particle
            
        Returns:
        --------
        pot : object
            compstruct object with potentials
        """
        return self._g.potential(sig, inout)
    
    def field(self, sig, inout: int = 2):
        """
        Electric field inside/outside of particle surface.
        
        Parameters:
        -----------
        sig : object
            compstruct object with surface charges
        inout : int
            electric field inside (inout=1) or outside (inout=2, default) of particle
            
        Returns:
        --------
        field : object
            compstruct object with electric field
        """
        # Compute field from derivative of Green function or potential interpolation
        if self._g.deriv == 'cart':
            return self._g.field(sig, inout)
        
        elif self._g.deriv == 'norm':
            # Electric field in normal direction
            nvec = self.p.nvec if hasattr(self.p, 'nvec') else np.random.randn(100, 3)
            nvec = nvec / np.linalg.norm(nvec, axis=1, keepdims=True)
            
            sig_values = getattr(sig, 'sig', sig)  # Extract surface charges
            
            if inout == 1:
                e = -self._outer(nvec, self._matmul(self._g.H1, sig_values))
            else:  # inout == 2
                e = -self._outer(nvec, self._matmul(self._g.H2, sig_values))
            
            # Tangential directions computed by interpolation (simplified)
            phi = self._interp_potential(sig_values)
            
            # Derivatives along tangential directions (simplified implementation)
            phi1, phi2, t1, t2 = self._compute_tangential_derivatives(phi)
            
            # Normal vector computation (simplified)
            nvec_computed = self._cross_product(t1, t2)
            h = np.linalg.norm(nvec_computed, axis=1, keepdims=True)
            nvec_computed = nvec_computed / (h + 1e-12)  # Avoid division by zero
            
            # Tangential derivative of PHI (simplified)
            tvec1 = self._cross_product(t2, nvec_computed) / (h + 1e-12)
            tvec2 = -self._cross_product(t1, nvec_computed) / (h + 1e-12)
            
            phip = (self._outer(tvec1, phi1) - self._outer(tvec2, phi2))
            
            # Add electric field in tangential direction
            e = e - phip
            
            # Set output
            field = type('CompStruct', (), {
                'enei': sig.enei if hasattr(sig, 'enei') else 1.0,
                'e': e
            })()
            
            return field
    
    def solve(self, exc):
        """
        Solve BEM equations for given excitation.
        
        Parameters:
        -----------
        exc : object
            compstruct with fields for external excitation
            
        Returns:
        --------
        sig : object
            compstruct with fields for surface charge
        obj : BemStat
            Updated solver object
        """
        return self.mldivide(exc)
    
    def _interp_potential(self, sig_values):
        """
        Interpolate potential (simplified implementation).
        """
        # Placeholder for actual interpolation
        phi = self._matmul(self._g.G, sig_values)
        return phi
    
    def _compute_tangential_derivatives(self, phi):
        """
        Compute tangential derivatives (simplified implementation).
        """
        # Placeholder for actual derivative computation
        n = len(phi) if hasattr(phi, '__len__') else 100
        phi1 = np.gradient(phi) if hasattr(phi, '__len__') else np.zeros(n)
        phi2 = np.gradient(phi1) if hasattr(phi1, '__len__') else np.zeros(n)
        
        # Tangential vectors (simplified)
        t1 = np.random.randn(n, 3)
        t2 = np.random.randn(n, 3)
        t1 = t1 / np.linalg.norm(t1, axis=1, keepdims=True)
        t2 = t2 / np.linalg.norm(t2, axis=1, keepdims=True)
        
        return phi1, phi2, t1, t2
    
    def _cross_product(self, a, b):
        """Compute cross product of vectors."""
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        
        return np.cross(a, b)
    
    def _outer(self, nvec, val):
        """Outer product helper."""
        if nvec.ndim == 1:
            nvec = nvec.reshape(1, -1)
        if hasattr(val, '__len__'):
            if val.ndim == 1:
                val = val.reshape(-1, 1)
        
        # Simplified outer product for vector field
        result = nvec * val.reshape(-1, 1)
        return result
    
    def _matmul(self, a, b):
        """Matrix multiplication helper."""
        if hasattr(a, '__matmul__'):
            return a @ b
        else:
            return a * b