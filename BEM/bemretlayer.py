"""
BEM Retarded Layer Solver - Python Implementation

BEM solver for full Maxwell equations and layer structure.
Given an external excitation, BEMRETLAYER computes the surface charges 
such that the boundary conditions of Maxwell's equations are fulfilled.

References: 
- Garcia de Abajo and Howie, PRB 65, 115418 (2002)
- Waxenegger et al., Comp. Phys. Commun. 193, 138 (2015)

Translated from MATLAB MNPBEM toolbox @bemretlayer class.
"""

import numpy as np
import warnings
from typing import Optional, Tuple, Any, Dict, Union
from ..Base.bembase import BemBase


class BemRetLayer(BemBase):
    """
    BEM solver for full Maxwell equations and layer structure.
    
    Given an external excitation, BEMRETLAYER computes the surface
    charges such that the boundary conditions of Maxwell's equations
    are fulfilled for systems with layered substrates.
    
    References: 
    - Garcia de Abajo and Howie, PRB 65, 115418 (2002)
    - Waxenegger et al., Comp. Phys. Commun. 193, 138 (2015)
    """
    
    # BemBase abstract properties
    name = 'bemsolver'
    needs = [{'sim': 'ret'}, 'layer']
    
    def __init__(self, *args, **kwargs):
        """
        Initialize BEM solver for layer system.
        
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
        self._npar = None       # parallel component of outer surface normal
        self._eps1 = None       # dielectric function inside of surface
        self._eps2 = None       # dielectric function outside of surface
        self._L1 = None         # G1e * inv(G1)
        self._L2p = None        # G2e * inv(G2), parallel component
        self._G1i = None        # inverse of inside Green function G1
        self._G2pi = None       # inverse of outside parallel Green function G2
        self._G2 = None         # outside Green function G2
        self._G2e = None        # G2 multiplied with dielectric function
        self._Sigma1 = None     # H1 * inv(G1), Eq. (21)
        self._Sigma1e = None    # H1e * inv(G1)
        self._Gamma = None      # inv(Sigma1 - Sigma2)
        self._m = None          # response matrix for layer system
        
        # Initialize
        self._init(*args, **kwargs)
    
    def __str__(self):
        """String representation."""
        return f"BemRetLayer(p={type(self.p).__name__ if self.p else None}, g={type(self.g).__name__ if self.g else None})"
    
    def __repr__(self):
        """Command window display."""
        return f"bemretlayer:\n{{'p': {self.p}, 'g': {self.g}}}"
    
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
        
        # Green function for layer structure
        self.g = self._compgreenretlayer(p, p, options)
        
        # Make sure that particle is not too close to interface
        if hasattr(self.g, 'layer') and hasattr(p, 'pos'):
            min_dist = self._mindist_layer(self.g.layer, p.pos[:, 2])
            if min_dist < 1e-5:
                warnings.warn(
                    'Particle might be too close to layer structure',
                    UserWarning
                )
        
        # Initialize for given wavelength
        if enei is not None:
            self._initmat(enei)
    
    def _compgreenretlayer(self, p1, p2, options):
        """
        Placeholder for layer Green function initialization.
        In actual implementation, this would create the layer retarded Green function.
        """
        # This would be implemented with actual layer Green function computation
        return type('LayerGreenFunction', (), {
            'G': lambda enei: self._create_placeholder_g_matrix(),
            'H1': lambda enei: np.eye(100),
            'H2': lambda enei: np.eye(100),
            'potential': lambda sig, inout: sig,
            'field': lambda sig, inout: sig,
            'deriv': 'cart',
            'layer': type('Layer', (), {'z': [0]})(),
            'initrefl': lambda enei: None
        })()
    
    def _create_placeholder_g_matrix(self):
        """Create placeholder G matrix structure for layer system."""
        return {
            'ss': np.eye(100),  # scalar-scalar
            'hh': np.eye(100),  # h-h component  
            'p': np.eye(100),   # parallel component
            'sh': np.zeros((100, 100)),  # scalar-h
            'hs': np.zeros((100, 100))   # h-scalar
        }
    
    def _mindist_layer(self, layer, z_positions):
        """Calculate minimum distance to layer interfaces."""
        if hasattr(layer, 'z'):
            layer_z = np.array(layer.z)
            min_distances = []
            for z in z_positions:
                min_distances.append(np.min(np.abs(z - layer_z)))
            return np.min(min_distances)
        return float('inf')
    
    def _initmat(self, enei: float):
        """
        Initialize matrices for BEM solver.
        
        Reference: Waxenegger et al., Comp. Phys. Commun. 193, 128 (2015).
        """
        # Use previously computed matrices?
        if not self.enei or self.enei != enei:
            # Save wavelength
            self.enei = enei
            
            # Initialize reflected Green functions
            if hasattr(self.g, 'initrefl'):
                self.g.initrefl(enei)
            
            # Wavelength of light in vacuum
            k = 2 * np.pi / enei
            
            # Dielectric functions
            eps1 = self.p.eps1(enei) if hasattr(self.p, 'eps1') else np.ones(100)
            eps2 = self.p.eps2(enei) if hasattr(self.p, 'eps2') else np.ones(100)
            
            # Simplify for unique dielectric functions
            if len(np.unique(eps1)) == 1 and len(np.unique(eps2)) == 1:
                eps1, eps2 = eps1[0], eps2[0]  # Note: fixed indexing
            else:
                eps1, eps2 = self._spdiag(eps1), self._spdiag(eps2)
            
            # Green functions for inner surfaces (placeholder implementation)
            G11 = self.g.G(enei)
            G21 = self.g.G(enei) 
            H11 = self.g.H1(enei)
            H21 = self.g.H1(enei)
            
            # Mixed contributions
            G1 = self._subtract_matrices(G11, G21)
            G1e = self._multiply_eps(eps1, G11, eps2, G21, subtract=True)
            H1 = self._subtract_matrices(H11, H21)
            H1e = self._multiply_eps(eps1, H11, eps2, H21, subtract=True)
            
            # Green functions for outer surfaces (placeholder implementation)
            G22 = self.g.G(enei)
            G12 = self.g.G(enei)
            H22 = self.g.H2(enei)
            H12 = self.g.H2(enei)
            
            # Mixed contributions for structured G2 and G2e
            G2 = self._create_structured_matrix(G22, G12, 'G')
            G2e = self._create_structured_matrix_eps(G22, G12, eps2, eps1, 'G')
            H2 = self._create_structured_matrix(H22, H12, 'H')
            H2e = self._create_structured_matrix_eps(H22, H12, eps2, eps1, 'H')
            
            # Auxiliary matrices
            # Inverse of G1 and of parallel component
            G1i = np.linalg.inv(G1)
            G2pi = np.linalg.inv(G2['p'])
            
            # Sigma matrices [Eq.(21)]
            Sigma1 = H1 @ G1i
            Sigma1e = H1e @ G1i
            Sigma2p = H2['p'] @ G2pi
            
            # Auxiliary dielectric function matrices
            L1 = G1e @ G1i
            L2p = G2e['p'] @ G2pi
            
            # Normal vectors
            nvec = self.p.nvec if hasattr(self.p, 'nvec') else np.random.randn(100, 3)
            nvec = nvec / np.linalg.norm(nvec, axis=1, keepdims=True)  # Normalize
            
            # Perpendicular and parallel component
            nperp = nvec[:, 2]
            npar = nvec - nperp[:, np.newaxis] * np.array([0, 0, 1])
            
            # Gamma matrix
            Gamma = np.linalg.inv(Sigma1 - Sigma2p)
            Gammapar = 1j * k * (L1 - L2p) @ Gamma * (npar @ npar.T)
            
            # Set up full matrix, Eq. (10)
            m = {}
            m[(1, 1)] = (Sigma1e @ G2['ss'] - H2e['ss'] - 1j * k * 
                        (Gammapar @ (L1 @ G2['ss'] - G2e['ss']) + 
                         np.outer(L1 @ G2['sh'] - G2e['sh'], nperp)))
            
            m[(1, 2)] = (Sigma1e @ G2['sh'] - H2e['sh'] - 1j * k * 
                        (Gammapar @ (L1 @ G2['sh'] - G2e['sh']) + 
                         np.outer(L1 @ G2['hh'] - G2e['hh'], nperp)))
            
            m[(2, 1)] = (Sigma1 @ G2['hs'] - H2['hs'] - 1j * k * 
                        np.outer(L1 @ G2['ss'] - G2e['ss'], nperp))
            
            m[(2, 2)] = (Sigma1 @ G2['hh'] - H2['hh'] - 1j * k * 
                        np.outer(L1 @ G2['sh'] - G2e['sh'], nperp))
            
            # Save matrices
            self._k = k
            self._nvec = nvec
            self._npar = npar
            self._eps1 = eps1
            self._eps2 = eps2
            self._L1 = L1
            self._L2p = L2p
            self._G1i = G1i
            self._G2pi = G2pi
            self._G2 = G2
            self._G2e = G2e
            self._Sigma1 = Sigma1
            self._Sigma1e = Sigma1e
            self._Gamma = Gamma
            self._m = m
    
    def _subtract_matrices(self, A, B):
        """Subtract two matrices, handling different types."""
        if isinstance(A, dict) and isinstance(B, dict):
            result = {}
            for key in A.keys():
                if key in B:
                    result[key] = A[key] - B[key]
                else:
                    result[key] = A[key]
            return result
        else:
            return A - B
    
    def _multiply_eps(self, eps1, A, eps2, B, subtract=False):
        """Multiply matrices with dielectric constants."""
        if hasattr(eps1, '__matmul__'):
            result1 = eps1 @ A
        else:
            result1 = eps1 * A
        
        if hasattr(eps2, '__matmul__'):
            result2 = eps2 @ B
        else:
            result2 = eps2 * B
        
        return result1 - result2 if subtract else result1 + result2
    
    def _create_structured_matrix(self, A, B, matrix_type='G'):
        """Create structured matrix for layer system."""
        if isinstance(A, dict):
            result = {}
            for key in ['ss', 'hh', 'p', 'sh', 'hs']:
                if key in A and key in B:
                    if key in ['sh', 'hs'] and matrix_type == 'G':
                        result[key] = A[key]  # Only A component for sh, hs
                    else:
                        result[key] = A[key] - B[key] if key in ['ss', 'hh', 'p'] else A[key]
                else:
                    result[key] = A if key in ['ss', 'hh', 'p'] else np.zeros_like(A)
            return result
        else:
            # For simple matrices, create the structure
            return {
                'ss': A - B,
                'hh': A - B, 
                'p': A - B,
                'sh': A,
                'hs': A
            }
    
    def _create_structured_matrix_eps(self, A, B, eps_a, eps_b, matrix_type='G'):
        """Create structured matrix with dielectric function multiplication."""
        if isinstance(A, dict):
            result = {}
            for key in ['ss', 'hh', 'p', 'sh', 'hs']:
                if key in A:
                    if key in ['sh', 'hs']:
                        result[key] = eps_a * A[key]  # Only eps_a * A for sh, hs
                    else:
                        A_eps = eps_a * A[key] if hasattr(eps_a, '__matmul__') else eps_a * A[key]
                        B_eps = eps_b * B[key] if hasattr(eps_b, '__matmul__') and key in B else 0
                        result[key] = A_eps - B_eps
                else:
                    result[key] = np.zeros_like(A)
            return result
        else:
            # For simple matrices
            A_eps = eps_a * A if hasattr(eps_a, '__matmul__') else eps_a * A
            B_eps = eps_b * B if hasattr(eps_b, '__matmul__') else eps_b * B
            return {
                'ss': A_eps - B_eps,
                'hh': A_eps - B_eps,
                'p': A_eps - B_eps,
                'sh': A_eps,
                'hs': A_eps
            }
    
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
        obj : BemRetLayer
            Updated object with initialized matrices
        """
        self._initmat(enei)
        return self
    
    def __truediv__(self, exc):
        """Surface charges and currents for given excitation."""
        return self.mldivide(exc)
    
    def mldivide(self, exc):
        """
        Surface charges and currents for given excitation.
        
        Reference: Waxenegger et al., Comp. Phys. Commun. 193, 138 (2015).
        
        Parameters:
        -----------
        exc : object
            compstruct with fields for external excitation
            
        Returns:
        --------
        sig : object
            compstruct with fields for surface charges and currents
        obj : BemRetLayer
            Updated solver object
        """
        # Initialize BEM solver (if needed)
        self._initmat(exc.enei)
        
        # Extract stored variables
        k = self._k
        nvec = self._nvec
        npar = self._npar
        L1 = self._L1
        L2p = self._L2p
        G1i = self._G1i
        G2pi = self._G2pi
        G2 = self._G2
        G2e = self._G2e
        Sigma1 = self._Sigma1
        Sigma1e = self._Sigma1e
        Gamma = self._Gamma
        m = self._m
        
        # Number of boundary elements
        n = self.p.n if hasattr(self.p, 'n') else 100
        
        # Unit vector in z-direction
        zunit = np.tile([0, 0, 1], (n, 1))
        
        # External excitation
        phi, a, alpha, De = self._excitation(exc)
        
        # Decompose vector potential into parallel and perpendicular components
        aperp = self._inner(zunit, a)
        apar = a - self._outer(zunit, aperp)
        
        # Solve BEM equations
        # Modify alpha and De, Eq. (9,10)
        alpha = (alpha - self._matmul(Sigma1, a) + 
                1j * k * self._outer(nvec, self._matmul(L1, phi)))
        
        De = (De - self._matmul(Sigma1e, phi) +
              1j * k * self._inner(nvec, self._matmul(L1, a)) +
              1j * k * self._inner(npar, self._matmul((L1 - L2p) @ Gamma, alpha)))
        
        # Decompose into parallel and perpendicular components
        alphaperp = self._inner(zunit, alpha)
        alphapar = alpha - self._outer(zunit, alphaperp)
        
        # Solve matrix equation, Eq. (10)
        m_matrix = np.block([[m[(1,1)], m[(1,2)]], [m[(2,1)], m[(2,2)]]])
        rhs = np.concatenate([De.flatten(), alphaperp.flatten()])
        xi2 = np.linalg.solve(m_matrix, rhs)
        
        # Decompose into surface charge and perpendicular surface current
        sig2 = xi2[:n].reshape(De.shape)
        h2perp = xi2[n:].reshape(De.shape)
        
        # Parallel component of Green function, Eq. (8)
        h2par = self._matmul(G2pi @ Gamma, alphapar +
                            1j * k * self._outer(npar, 
                                               self._matmul(L1 @ G2['ss'] - G2e['ss'], sig2) +
                                               self._matmul(L1 @ G2['sh'] - G2e['sh'], h2perp)))
        
        # Surface current
        h2 = h2par + self._outer(zunit, h2perp)
        
        # Surface charges at inner interface
        sig1 = self._matmul(G1i, 
                           self._matmul(G2['ss'], sig2) + 
                           self._matmul(G2['sh'], h2perp) + phi)
        
        # Surface currents at inner interface
        h1perp = self._matmul(G1i, 
                             self._matmul(G2['hs'], sig2) + 
                             self._matmul(G2['hh'], h2perp) + aperp)
        h1par = self._matmul(G1i, self._matmul(G2['p'], h2par) + apar)
        
        # Surface current
        h1 = h1par + self._outer(zunit, h1perp)
        
        # Save everything in single structure
        sig = type('CompStruct', (), {
            'enei': exc.enei,
            'sig1': sig1,
            'sig2': sig2,
            'h1': h1,
            'h2': h2
        })()
        
        return sig, self
    
    def potential(self, sig, inout: int = 2):
        """
        Potentials and surface derivatives inside/outside of particle.
        
        Parameters:
        -----------
        sig : object
            compstruct with surface charges
        inout : int
            potential inside (inout=1) or outside (inout=2, default) of particle
            
        Returns:
        --------
        pot : object
            compstruct object with potentials
        """
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
            # Wavenumber of light in vacuum
            k = 2 * np.pi / sig.enei
            
            # Potential
            pot = self.potential(sig, inout)
            
            # Extract fields (simplified implementation)
            if hasattr(pot, 'phi1'):
                phi, phip, a, ap = pot.phi1, pot.phi1p, pot.a1, pot.a1p
            else:
                phi, phip, a, ap = pot.phi2, pot.phi2p, pot.a2, pot.a2p
            
            # Simplified field calculation (placeholder)
            e = 1j * k * a  # Simplified
            h = a  # Simplified
            
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
        obj : BemRetLayer
            Updated solver object
        """
        return self.mldivide(exc)
    
    def _excitation(self, exc):
        """Compute excitation variables for BEM solver."""
        # Default values for scalar and vector potential
        phi1, phi1p, a1, a1p = 0, 0, 0, 0
        phi2, phi2p, a2, a2p = 0, 0, 0, 0
        
        # Get input fields (simplified)
        if hasattr(exc, 'phi1'):
            phi1 = exc.phi1
        if hasattr(exc, 'phi2'):
            phi2 = exc.phi2
        # ... similar for other fields
        
        # Wavenumber and dielectric functions
        k = self._k
        eps1 = self._eps1
        eps2 = self._eps2
        
        # Outer surface normal
        nvec = self._nvec
        
        # External excitation, Garcia de Abajo and Howie, PRB 65, 115418 (2002)
        # Eqs. (10,11)
        phi = phi2 - phi1
        a = a2 - a1
        
        # Eq. (15)
        alpha = (a2p - a1p - 
                1j * k * (self._outer(nvec, phi2, eps2) - 
                         self._outer(nvec, phi1, eps1)))
        
        # Eq. (18)
        De = (self._matmul(eps2, phi2p) - self._matmul(eps1, phi1p) -
              1j * k * (self._inner(nvec, a2, eps2) - 
                       self._inner(nvec, a1, eps1)))
        
        # Expand PHI and A
        if np.isscalar(phi) and phi == 0:
            phi = np.zeros_like(De)
        if np.isscalar(a) and a == 0:
            a = np.zeros_like(alpha)
        
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
            return np.sum(a * b * c, axis=-1)
        else:
            return np.sum(a * b, axis=-1)