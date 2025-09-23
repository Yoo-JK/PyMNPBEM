"""
BEM Retarded Solver - Python Implementation

BEM solver for full Maxwell equations.
Given an external excitation, BEMRET computes the surface charges 
such that the boundary conditions of Maxwell's equations are fulfilled.

See, e.g. Garcia de Abajo and Howie, PRB 65, 115418 (2002).
Translated from MATLAB MNPBEM toolbox @bemret class.
"""

import numpy as np
from typing import Optional, Tuple, Any, Dict, Union
from ..Base.bembase import BemBase


class BemRet(BemBase):
    """
    BEM solver for full Maxwell equations.
    
    Given an external excitation, BEMRET computes the surface
    charges such that the boundary conditions of Maxwell's equations
    are fulfilled.
    
    Reference: Garcia de Abajo and Howie, PRB 65, 115418 (2002).
    """
    
    # BemBase abstract properties
    name = 'bemsolver'
    needs = [{'sim': 'ret'}]
    
    def __init__(self, *args, **kwargs):
        """
        Initialize BEM solver.
        
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
        self.p = None           # compound of discretized particles
        self.g = None           # Green function
        self.enei = []          # light wavelength in vacuum
        
        # Private properties
        self._k = None          # wavenumber of light in vacuum
        self._nvec = None       # outer surface normals of discretized surface
        self._eps1 = None       # dielectric function inside of surface
        self._eps2 = None       # dielectric function outside of surface
        self._G1i = None        # inverse of inside Green function G1
        self._G2i = None        # inverse of outside Green function G2
        self._L1 = None         # G1 * eps1 * G1i, Eq. (22)
        self._L2 = None         # G2 * eps2 * G2i
        self._Sigma1 = None     # H1 * G1i, Eq. (21)
        self._Deltai = None     # inv(Sigma1 - Sigma2)
        self._Sigmai = None     # Eq. (21,22)
        
        # Initialize
        self._init(*args, **kwargs)
    
    def __str__(self):
        """String representation."""
        return f"BemRet(p={type(self.p).__name__ if self.p else None}, g={type(self.g).__name__ if self.g else None})"
    
    def __repr__(self):
        """Command window display."""
        return f"bemret:\n{{'p': {self.p}, 'g': {self.g}}}"
    
    def _init(self, p, *args, **kwargs):
        """
        Initialize full BEM solver.
        
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
        
        # Green function - placeholder for actual implementation
        self.g = self._compgreenret(p, p, options)
        
        # Initialize for given wavelength
        if enei is not None:
            self._initmat(enei)
    
    def _compgreenret(self, p1, p2, options):
        """
        Placeholder for Green function initialization.
        In actual implementation, this would create the retarded Green function.
        """
        # This would be implemented with actual Green function computation
        return type('GreenFunction', (), {
            'G': lambda enei: np.eye(100),  # Placeholder
            'H1': lambda enei: np.eye(100),  # Placeholder
            'H2': lambda enei: np.eye(100),  # Placeholder
            'potential': lambda sig, inout: sig,  # Placeholder
            'field': lambda sig, inout: sig,  # Placeholder
            'deriv': 'cart',
            'con': {(1, 2): np.zeros((100, 100))}  # Placeholder
        })()
    
    def clear(self):
        """
        Clear Green functions and auxiliary matrices.
        
        Returns:
        --------
        obj : BemRet
            Updated object with cleared matrices
        """
        self._G1i = None
        self._G2i = None
        self._L1 = None
        self._L2 = None
        self._Sigma1 = None
        self._Deltai = None
        self._Sigmai = None
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
        if not self.enei or self.enei != enei:
            self.enei = enei
            
            # Auxiliary quantities
            nvec = self.p.nvec if hasattr(self.p, 'nvec') else np.eye(3)[:100]  # Placeholder
            k = 2 * np.pi / enei
            
            # Dielectric functions
            eps1 = self._spdiag(self.p.eps1(enei) if hasattr(self.p, 'eps1') else 1.0)
            eps2 = self._spdiag(self.p.eps2(enei) if hasattr(self.p, 'eps2') else 1.0)
            
            # Simplify for unique dielectric functions
            if hasattr(eps1, 'diagonal'):
                if len(np.unique(eps1.diagonal())) == 1 and len(np.unique(eps2.diagonal())) == 1:
                    eps1 = float(eps1.diagonal()[0])
                    eps2 = float(eps2.diagonal()[0])
            
            # Green functions and surface derivatives (placeholder implementation)
            G1 = self.g.G(enei) - self.g.G(enei)  # Placeholder
            G1i = np.linalg.inv(G1)
            G2 = self.g.G(enei) - self.g.G(enei)  # Placeholder
            G2i = np.linalg.inv(G2)
            H1 = self.g.H1(enei) - self.g.H1(enei)  # Placeholder
            H2 = self.g.H2(enei) - self.g.H2(enei)  # Placeholder
            
            # L matrices [Eq. (22)]
            if np.all(self.g.con[(1, 2)] == 0):
                L1 = eps1
                L2 = eps2
            else:
                L1 = G1 @ eps1 @ G1i if hasattr(eps1, 'shape') else eps1 * G1 @ G1i
                L2 = G2 @ eps2 @ G2i if hasattr(eps2, 'shape') else eps2 * G2 @ G2i
            
            # Sigma and Delta matrices, and combinations
            Sigma1 = H1 @ G1i
            Sigma2 = H2 @ G2i
            
            # Inverse Delta matrix
            Deltai = np.linalg.inv(Sigma1 - Sigma2)
            
            # Difference of dielectric functions
            L = L1 - L2
            
            # Sigma matrix
            if hasattr(L, 'shape'):
                nvec_outer = nvec @ nvec.T
                Sigma = (Sigma1 @ L1 - Sigma2 @ L2 + 
                        k**2 * (L @ Deltai * nvec_outer) @ L)
            else:
                nvec_outer = nvec @ nvec.T
                Sigma = (Sigma1 * L1 - Sigma2 * L2 + 
                        k**2 * L * (Deltai * nvec_outer) * L)
            
            # Save everything in class
            self._k = k
            self._nvec = nvec
            self._eps1 = eps1
            self._eps2 = eps2
            self._G1i = G1i
            self._G2i = G2i
            self._L1 = L1
            self._L2 = L2
            self._Sigma1 = Sigma1
            self._Deltai = Deltai
            self._Sigmai = np.linalg.inv(Sigma)
    
    def _spdiag(self, values):
        """Create sparse diagonal matrix or return scalar."""
        if hasattr(values, '__len__') and len(values) > 1:
            return np.diag(values)
        else:
            return values
    
    def __call__(self, enei: float):
        """
        Initialize matrices for given energy.
        
        Parameters:
        -----------
        enei : float
            Light wavelength in vacuum
            
        Returns:
        --------
        obj : BemRet
            Updated object with initialized matrices
        """
        self._initmat(enei)
        return self
    
    def __truediv__(self, exc):
        """
        Surface charges and currents for given excitation.
        
        Parameters:
        -----------
        exc : object
            compstruct with fields for external excitation
            
        Returns:
        --------
        sig : object
            compstruct with fields for surface charges and currents
        obj : BemRet
            Updated solver object
        """
        return self.mldivide(exc)
    
    def mldivide(self, exc):
        """
        Surface charges and currents for given excitation.
        
        Parameters:
        -----------
        exc : object
            compstruct with fields for external excitation
            
        Returns:
        --------
        sig : object
            compstruct with fields for surface charges and currents
        obj : BemRet
            Updated solver object
        """
        # Initialize BEM solver (if needed)
        self._initmat(exc.enei)
        
        # External perturbation and extract stored variables
        # Garcia de Abajo and Howie, PRB 65, 115418 (2002).
        phi, a, alpha, De = self._excitation(exc)
        
        k = self._k
        nvec = self._nvec
        G1i = self._G1i
        G2i = self._G2i
        L1 = self._L1
        L2 = self._L2
        Sigma1 = self._Sigma1
        Deltai = self._Deltai
        Sigmai = self._Sigmai
        
        # Solve BEM equations
        # Modify alpha and De
        alpha = (alpha - self._matmul(Sigma1, a) + 
                1j * k * self._outer(nvec, self._matmul(L1, phi)))
        De = (De - self._matmul(Sigma1, self._matmul(L1, phi)) + 
              1j * k * self._inner(nvec, self._matmul(L1, a)))
        
        # Eq. (19)
        sig2 = self._matmul(Sigmai, 
                           De + 1j * k * self._inner(nvec, self._matmul(
                               L1 - L2, self._matmul(Deltai, alpha))))
        
        # Eq. (20)
        h2 = self._matmul(Deltai, 
                         1j * k * self._outer(nvec, self._matmul(L1 - L2, sig2)) + alpha)
        
        # Surface charges and currents
        sig1 = self._matmul(G1i, sig2 + phi)
        h1 = self._matmul(G1i, h2 + a)
        sig2 = self._matmul(G2i, sig2)
        h2 = self._matmul(G2i, h2)
        
        # Save everything in single structure (placeholder)
        sig = type('CompStruct', (), {
            'enei': exc.enei,
            'sig1': sig1,
            'sig2': sig2,
            'h1': h1,
            'h2': h2
        })()
        
        return sig, self
    
    def __mul__(self, sig):
        """
        Induced potential for given surface charge.
        
        Parameters:
        -----------
        sig : object
            compstruct with fields for surface charges and currents
            
        Returns:
        --------
        phi : object
            compstruct with fields for induced potential
        """
        return self.potential(sig, 1) + self.potential(sig, 2)
    
    def potential(self, sig, inout: int = 2):
        """
        Determine potentials and surface derivatives inside/outside of particle.
        
        Parameters:
        -----------
        sig : object
            compstruct with surface charges (see bemret.mldivide)
        inout : int
            potential inside (inout=1) or outside (inout=2, default) of particle
            
        Returns:
        --------
        pot : object
            compstruct object with potentials
        """
        # Compute potential
        return self.g.potential(sig, inout)
    
    def field(self, sig, inout: int = 2):
        """
        Electric and magnetic field inside/outside of particle surface.
        
        Parameters:
        -----------
        sig : object
            compstruct object with surface charges
        inout : int
            electric field inside (inout=1) or outside (inout=2, default) of particle
            
        Returns:
        --------
        field : object
            compstruct object with electric and magnetic fields
        """
        # Compute field from derivative of Green function or from potential interpolation
        if self.g.deriv == 'cart':
            return self.g.field(sig, inout)
        
        elif self.g.deriv == 'norm':
            # One can additionally compute the electric fields using only the
            # surface derivatives and interpolating the potentials...
            k = 2 * np.pi / sig.enei
            
            # Potential
            pot = self.potential(sig, inout)
            
            # Extract fields (placeholder implementation)
            if hasattr(pot, 'phi1'):
                phi, phip, a, ap = pot.phi1, pot.phi1p, pot.a1, pot.a1p
            else:
                phi, phip, a, ap = pot.phi2, pot.phi2p, pot.a2, pot.a2p
            
            # Simplified field calculation (placeholder)
            e = 1j * k * a  # Simplified
            h = a  # Simplified
            
            # Set output (placeholder)
            field = type('CompStruct', (), {
                'enei': sig.enei,
                'e': e,
                'h': h
            })()
            
            return field
    
    def solve(self, exc, *args):
        """
        Compute surface charges and currents for given excitation.
        
        Parameters:
        -----------
        exc : object
            compstruct with fields for external excitation
            
        Returns:
        --------
        sig : object
            compstruct with fields for surface charges and currents
        obj : BemRet
            Updated solver object
        """
        return self.mldivide(exc)
    
    def _excitation(self, exc):
        """
        Compute excitation variables for BEM solver.
        
        Parameters:
        -----------
        exc : object
            External excitation object
            
        Returns:
        --------
        phi : array
            Potential difference
        a : array
            Vector potential difference
        alpha : array
            Alpha parameter
        De : array
            De parameter
        """
        # Default values for potentials
        phi1, phi1p, a1, a1p = 0, 0, 0, 0
        phi2, phi2p, a2, a2p = 0, 0, 0, 0
        
        # Extract fields (simplified - would need actual field extraction)
        if hasattr(exc, 'phi1'):
            phi1 = exc.phi1
        if hasattr(exc, 'phi2'):
            phi2 = exc.phi2
        # ... similar for other fields
        
        # Wavenumber of light in vacuum
        k = 2 * np.pi / self.enei
        
        # Dielectric functions
        eps1 = self.p.eps1(self.enei) if hasattr(self.p, 'eps1') else 1.0
        eps2 = self.p.eps2(self.enei) if hasattr(self.p, 'eps2') else 1.0
        
        # Outer surface normal
        nvec = self._nvec
        
        # External excitation - Garcia de Abajo and Howie, PRB 65, 115418 (2002)
        # Eqs. (10,11)
        phi = phi2 - phi1
        a = a2 - a1
        
        # Eq. (15) - simplified
        alpha = (a2p - a1p - 
                1j * k * (self._outer(nvec, phi2, eps2) - self._outer(nvec, phi1, eps1)))
        
        # Eq. (18) - simplified
        De = (self._matmul(eps2, phi2p) - self._matmul(eps1, phi1p) -
              1j * k * (self._inner(nvec, a2, eps2) - self._inner(nvec, a1, eps1)))
        
        return phi, a, alpha, De
    
    def _matmul(self, a, b):
        """Matrix multiplication helper."""
        if hasattr(a, '__matmul__'):
            return a @ b
        else:
            return a * b
    
    def _outer(self, a, b, c=None):
        """Outer product helper."""
        if c is not None:
            return np.outer(a, b * c)
        else:
            return np.outer(a, b)
    
    def _inner(self, a, b, c=None):
        """Inner product helper."""
        if c is not None:
            return np.inner(a, b * c)
        else:
            return np.inner(a, b)