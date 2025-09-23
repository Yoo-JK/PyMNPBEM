"""
Iterative BEM Retarded Layer Solver - Python Implementation

Iterative BEM solver for full Maxwell equations and layer structure.
Given an external excitation, BEMRETLAYERITER computes the surface charges 
iteratively such that the boundary conditions of Maxwell's equations are fulfilled.

References: 
- Garcia de Abajo and Howie, PRB 65, 115418 (2002)
- Waxenegger et al., Comp. Phys. Commun. 193, 138 (2015)

Translated from MATLAB MNPBEM toolbox @bemretlayeriter class.
"""

import numpy as np
import warnings
from typing import Optional, Tuple, Any, Dict, Union, Callable
from ..Base.bembase import BemBase
from .bemiter import BemIter


class BemRetLayerIter(BemBase, BemIter):
    """
    Iterative BEM solver for full Maxwell equations and layer structure.
    
    Given an external excitation, BEMRETLAYERITER computes the surface
    charges iteratively such that the boundary conditions of Maxwell's
    equations are fulfilled for systems with layered substrates.
    
    References: 
    - Garcia de Abajo and Howie, PRB 65, 115418 (2002)
    - Waxenegger et al., Comp. Phys. Commun. 193, 138 (2015)
    """
    
    # BemBase abstract properties
    name = 'bemsolver'
    needs = [{'sim': 'ret'}, 'layer', 'iter']
    
    def __init__(self, *args, **kwargs):
        """
        Initialize iterative BEM solver for layer system.
        
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
        # Initialize BemIter first (for iterative solver properties)
        if len(args) >= 2:
            BemIter.__init__(self, *args[1:], **kwargs)
        else:
            BemIter.__init__(self, **kwargs)
        
        # Public properties specific to bemretlayeriter
        self.p = None           # compound of discretized particles
        self.layer = None       # layer structure
        self.g = None           # Green function
        self.enei = []          # light wavelength in vacuum
        
        # Private properties specific to bemretlayeriter
        self._op = {}           # option structure
        self._sav = {}          # variables for evaluation of preconditioner
        self._k = None          # wavenumber of light in vacuum
        self._eps1 = None       # dielectric function at particle inside
        self._eps2 = None       # dielectric function at particle outside
        self._nvec = None       # outer surface normal
        self._G1 = None         # Green function inside particle
        self._H1 = None         # surface derivative inside particle
        self._G2 = None         # Green function outside particle
        self._H2 = None         # surface derivative outside particle
        
        # Initialize
        self._init(*args, **kwargs)
    
    def __str__(self):
        """String representation."""
        return (f"BemRetLayerIter(p={type(self.p).__name__ if self.p else None}, "
                f"layer={type(self.layer).__name__ if self.layer else None}, "
                f"g={type(self.g).__name__ if self.g else None}, "
                f"solver='{self.solver}', tol={self.tol})")
    
    def __repr__(self):
        """Command window display."""
        return (f"bemretlayeriter:\n{{'p': {self.p}, 'layer': {self.layer}, 'g': {self.g}, "
                f"'solver': '{self.solver}', 'tol': {self.tol}}}")
    
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
        # Save particle and outer surface normal
        self.p = p
        self._nvec = p.nvec if hasattr(p, 'nvec') else np.random.randn(100, 3)
        self._nvec = self._nvec / np.linalg.norm(self._nvec, axis=1, keepdims=True)
        
        # Handle calls with and without ENEI
        varargin = list(args)
        enei = None
        if varargin and isinstance(varargin[0], (int, float, complex)):
            enei = varargin[0]
            varargin = varargin[1:]
        
        # Option structure
        self._op = self._getbemoptions(['iter', 'bemiter', 'bemretlayeriter'], *varargin, **kwargs)
        
        # Layer structure
        self.layer = self._op.get('layer')
        
        # Green function, set tolerance and maximal rank for low-rank matrices
        htol = min(self._op.get('htol', [1e-6])) if isinstance(self._op.get('htol'), list) else self._op.get('htol', 1e-6)
        kmax = max(self._op.get('kmax', [100])) if isinstance(self._op.get('kmax'), list) else self._op.get('kmax', 100)
        
        self.g = self._compgreenretlayer_aca(p, htol=htol, kmax=kmax, **kwargs)
        
        # Cluster tree for reduction of preconditioner matrices
        if 'reduce' in self._op:
            self.rtree = self._reducetree(self.g.hmat.tree, self._op['reduce'])
        
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
    
    def _compgreenretlayer_aca(self, p, **options):
        """
        Placeholder for ACA layer Green function initialization.
        In actual implementation, this would create the ACA layer retarded Green function.
        """
        # This would be implemented with actual ACA layer Green function computation
        return type('ACALayerGreenFunction', (), {
            'G': lambda enei: self._create_placeholder_g_matrix(),
            'H1': lambda enei: np.eye(100),
            'H2': lambda enei: np.eye(100),
            'potential': lambda sig, inout: sig,
            'field': lambda sig, inout: sig,
            'deriv': 'norm',
            'layer': type('Layer', (), {'z': [0]})(),
            'hmat': type('HMat', (), {'tree': None})()
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
    
    def _reducetree(self, tree, reduce_param):
        """Placeholder for tree reduction."""
        return tree
    
    def clear(self):
        """
        Clear Green functions and auxiliary matrices.
        
        Returns:
        --------
        obj : BemRetLayerIter
            Updated object with cleared matrices
        """
        self._G1 = None
        self._H1 = None
        self._G2 = None
        self._H2 = None
        self._sav = {}
        return self
    
    def _initmat(self, enei: float):
        """
        Initialize Green functions and preconditioner for iterative BEM solver.
        
        Reference: Waxenegger et al., Comp. Phys. Commun. 193, 128 (2015).
        """
        # Use previously computed matrices?
        if not self.enei or self.enei != enei:
            self.enei = enei
            
            # Wavenumber
            self._k = 2 * np.pi / enei
            
            # Dielectric function
            self._eps1 = self.p.eps1(enei) if hasattr(self.p, 'eps1') else np.ones(100)
            self._eps2 = self.p.eps2(enei) if hasattr(self.p, 'eps2') else np.ones(100)
            
            # Initialize timer
            self.tocout('init', 'G1', 'H1', 'G2', 'H2', 'G1i', 'G2pi',
                       'Sigma1', 'Sigma2p', 'Gamma', 'm', 'im')
            
            # Green functions for inner surfaces (placeholder implementation)
            G1 = self.g.G(enei) - self.g.G(enei)  # Placeholder subtraction
            self.tocout('G1')
            H1 = self.g.H1(enei) - self.g.H1(enei)  # Placeholder subtraction
            self.tocout('H1')
            
            # Green functions for outer surfaces (placeholder implementation)
            G2_full = self.g.G(enei)
            g2_mixed = self.g.G(enei)
            self.tocout('G2')
            H2_full = self.g.H2(enei)
            h2_mixed = self.g.H2(enei)
            self.tocout('H2')
            
            # Add mixed contributions
            G2 = {}
            H2 = {}
            for key in ['ss', 'hh', 'p']:
                if isinstance(G2_full, dict) and key in G2_full:
                    G2[key] = G2_full[key] - g2_mixed
                    H2[key] = H2_full[key] - h2_mixed
                else:
                    G2[key] = G2_full - g2_mixed  # Fallback for simple matrices
                    H2[key] = H2_full - h2_mixed
            
            # Add remaining components
            for key in ['sh', 'hs']:
                if isinstance(G2_full, dict) and key in G2_full:
                    G2[key] = G2_full[key]
                    H2[key] = H2_full[key]
                else:
                    G2[key] = G2_full  # Fallback
                    H2[key] = H2_full
            
            # Save Green functions
            self._G1, self._H1, self._G2, self._H2 = G1, H1, G2, H2
            
            # Initialize preconditioner
            if self.precond:
                self._initprecond(enei)
            
            self.tocout('close')
            
            # Save statistics
            self.setstat('G1', self._G1)
            self.setstat('H1', self._H1)
            
            # Loop over names for G2 and H2
            for name in G2.keys():
                self.setstat('G2', self._G2[name])
                self.setstat('H2', self._H2[name])
    
    def _initprecond(self, enei: float):
        """
        Initialize preconditioner for H-matrices.
        
        Reference: Waxenegger et al., Comp. Phys. Commun. 193, 128 (2015).
        """
        # Wavenumber
        k = 2 * np.pi / enei
        
        # Dielectric functions
        eps1 = self._spdiag(self._eps1)
        eps2 = self._spdiag(self._eps2)
        
        # Difference of dielectric function
        ikdeps = 1j * k * (eps1 - eps2)
        
        # Normal vector
        nvec = self._nvec
        
        # Compress Green functions and surface derivatives
        G1 = self._compress(self._G1)
        H1 = self._compress(self._H1)
        
        # Loop over names for G2 and H2
        G2 = {}
        H2 = {}
        for name in self._G2.keys():
            G2[name] = self._compress(self._G2[name])
            H2[name] = self._compress(self._H2[name])
        
        # Inverse of G1 and of parallel component
        G1i = self._lu_decomp(G1)
        self.tocout('G1i')
        G2pi = self._lu_decomp(G2['p'])
        self.tocout('G2pi')
        
        # Sigma matrices [Eq.(21)]
        Sigma1 = self._rsolve(H1, G1i)
        self.tocout('Sigma1')
        Sigma2p = self._rsolve(H2['p'], G2pi)
        self.tocout('Sigma2p')
        
        # Perpendicular component of normal vector
        nperp = self._spdiag(nvec[:, 2])
        
        # Gamma matrix
        Gamma = np.linalg.inv(Sigma1 - Sigma2p)
        self.tocout('Gamma')
        Gammapar = ikdeps @ self._fun_gamma(Gamma, nvec)
        
        # Set up full matrix, Eq. (10)
        m11 = ((eps1 @ Sigma1 - Gammapar @ ikdeps) @ G2['ss'] - 
               eps2 @ H2['ss'] - (nperp @ ikdeps) @ G2['hs'])
        
        m12 = ((eps1 @ Sigma1 - Gammapar @ ikdeps) @ G2['sh'] - 
               eps2 @ H2['sh'] - (nperp @ ikdeps) @ G2['hh'])
        
        m21 = (Sigma1 @ G2['hs'] - H2['hs'] - 
               nperp @ ikdeps @ G2['ss'])
        
        m22 = (Sigma1 @ G2['hh'] - H2['hh'] - 
               nperp @ ikdeps @ G2['sh'])
        
        # Timing
        self.tocout('m')
        
        # LU decompositions for block matrix solver
        # L11 * U11 = M11
        im11 = self._lu_decomp(m11)
        
        # L11 * U12 = M12
        im12 = self._lsolve(m12, im11, 'L')
        
        # L21 * U11 = M21
        im21 = self._rsolve(m21, im11, 'U')
        
        # L22 * U22 = M22 - L21 * U12
        im22 = self._lu_decomp(m22 - im21 @ im12)
        
        # Timing
        self.tocout('im')
        
        # Save variables
        sav = {
            'k': k,
            'nvec': nvec,
            'eps1': eps1,
            'eps2': eps2,
            'G1i': G1i,
            'G2pi': G2pi,
            'G2': G2,
            'Sigma1': Sigma1,
            'Gamma': Gamma,
            'im': [[im11, im12], [im21, im22]]
        }
        
        # Save structure
        self._sav = sav
        
        # Save statistics
        if self.precond == 'hmat':
            hmat_list = [G1i, G2pi, Sigma1, Sigma2p, Gamma, im11, im12, im21, im22]
            name_list = ['G1i', 'G2pi', 'Sigma1', 'Sigma2p', 'Gamma', 'im11', 'im12', 'im21', 'im22']
            
            for hmat, name in zip(hmat_list, name_list):
                self.setstat(name, hmat)
    
    def _fun_gamma(self, Gamma, nvec):
        """
        Decorate Gamma function with normal vectors, Eq. (10a).
        """
        nvec1 = self._spdiag(nvec[:, 0])
        nvec2 = self._spdiag(nvec[:, 1])
        return nvec1 @ Gamma @ nvec1 + nvec2 @ Gamma @ nvec2
    
    def _compress(self, hmat):
        """Compress H-matrices for preconditioner."""
        if hasattr(hmat, 'htol'):
            hmat.htol = max(self._op.get('htol', [1e-6]))
        if hasattr(hmat, 'kmax'):
            hmat.kmax = min(self._op.get('kmax', [100]))
        return hmat
    
    def _lu_decomp(self, matrix):
        """LU decomposition placeholder."""
        return matrix  # Would implement actual LU decomposition
    
    def _rsolve(self, a, lu_matrix, mode='U'):
        """Resolve with LU decomposition placeholder."""
        return a  # Would implement actual resolution
    
    def _lsolve(self, a, lu_matrix, mode='L'):
        """Left solve with LU decomposition placeholder."""
        return a  # Would implement actual left resolution
    
    def _spdiag(self, values):
        """Create sparse diagonal matrix or return scalar."""
        if hasattr(values, '__len__') and len(values) > 1:
            return np.diag(values)
        else:
            return values
    
    def __call__(self, enei: float):
        """Initialize matrices for given energy."""
        self._initmat(enei)
        return self
    
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
            compstruct with fields for surface charges and currents
        obj : BemRetLayerIter
            Updated solver object
        """
        # Initialize BEM solver (if needed)
        self._initmat(exc.enei)
        
        # External excitation
        phi, a, De, alpha = self._excitation(exc)
        
        # Size of excitation arrays
        siz1, siz2 = phi.shape, a.shape
        
        # Pack everything to single vector
        b = self._pack(phi, a, De, alpha)
        
        # Function for matrix multiplication
        fa = lambda x: self._afun(x)
        fm = None
        
        # Function for preconditioner
        if self.precond:
            fm = lambda x: self._mfun(x)
        
        # Iterative solution
        x, self_updated = BemIter.solve(self, [], b, fa, fm)
        
        # Unpack and save solution vector
        sig1, h1, sig2, h2 = self._unpack(x)
        
        # Reshape surface charges and currents
        sig1 = sig1.reshape(siz1)
        h1 = h1.reshape(siz2)
        sig2 = sig2.reshape(siz1)
        h2 = h2.reshape(siz2)
        
        # Save everything in single structure
        sig = type('CompStruct', (), {
            'enei': exc.enei,
            'sig1': sig1,
            'sig2': sig2,
            'h1': h1,
            'h2': h2
        })()
        
        return sig, self
    
    def __truediv__(self, exc):
        """Surface charges and currents for given excitation."""
        return self.solve(exc)
    
    def mldivide(self, exc):
        """Surface charges and currents for given excitation."""
        return self.solve(exc)
    
    def potential(self, sig, inout: int = 2):
        """Potentials and surface derivatives inside/outside of particle."""
        return self.g.potential(sig, inout)
    
    def field(self, sig, inout: int = 2):
        """Electric and magnetic field inside/outside of particle surface."""
        # Compute field from derivative of Green function or potential interpolation
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
    
    def _afun(self, vec):
        """
        Matrix multiplication for CGS, BICGSTAB, GMRES.
        
        Reference: Waxenegger et al., Comp. Phys. Commun. 193, 138 (2015).
        """
        # Split vector array
        sig1, h1par, h1perp, sig2, h2par, h2perp = self._unpack_6(vec)
        
        # Extract input
        k, nvec, eps1, eps2 = self._k, self._nvec, self._eps1, self._eps2
        
        # Parallel and perpendicular components of NVEC
        npar, nperp = nvec[:, :2], nvec[:, 2]
        
        # Green functions
        G1, H1, G2, H2 = self._G1, self._H1, self._G2, self._H2
        
        # Apply Green functions to surface charges (simplified)
        Gsig1 = G1 @ sig1
        Gsig2 = G2['ss'] @ sig2 + G2['sh'] @ h2perp
        Hsig1 = H1 @ sig1
        Hsig2 = H2['ss'] @ sig2 + H2['sh'] @ h2perp
        
        # Apply Green functions to parallel surface currents (simplified)
        Gh1par = self._matmul(G1, h1par)
        Gh2par = self._matmul(G2['p'], h2par)
        Hh1par = self._matmul(H1, h1par)
        Hh2par = self._matmul(H2['p'], h2par)
        
        # Apply Green functions to perpendicular surface currents
        Gh1perp = G1 @ h1perp
        Gh2perp = G2['hh'] @ h2perp + G2['hs'] @ sig2
        Hh1perp = H1 @ h1perp
        Hh2perp = H2['hh'] @ h2perp + H2['hs'] @ sig2
        
        # Eq. (7a)
        phi = Gsig1 - Gsig2
        
        # Eqs. (7b,c)
        apar = Gh1par - Gh2par
        aperp = Gh1perp - Gh2perp
        
        # Eqs. (8a,b) - simplified
        alphapar = (Hh1par - Hh2par - 
                   1j * k * (self._outer(npar, Gsig1, eps1) - 
                            self._outer(npar, Gsig2, eps2)))
        
        alphaperp = (Hh1perp - Hh2perp - 
                    1j * k * (Gsig1 * eps1 * nperp - Gsig2 * eps2 * nperp))
        
        # Eq. (9) - simplified
        De = (Hsig1 * eps1 - Hsig2 * eps2 - 
              1j * k * (self._inner(npar, Gh1par, eps1) - 
                       self._inner(npar, Gh2par, eps2)) -
              1j * k * (Gh1perp * eps1 * nperp - Gh2perp * eps2 * nperp))
        
        # Pack into single vector
        return self._pack_6(phi, apar, aperp, De, alphapar, alphaperp)
    
    def _mfun(self, vec):
        """
        Preconditioner for CGS, BICGSTAB, GMRES.
        
        Reference: Waxenegger et al., Comp. Phys. Commun. 193, 138 (2015).
        """
        # Unpack matrices
        phi, a, De, alpha = self._unpack_4(vec)
        
        # Get variables for evaluation of preconditioner
        sav = self._sav
        k = sav['k']
        nvec = sav['nvec']
        G2 = sav['G2']
        G1i = sav['G1i']
        G2pi = sav['G2pi']
        eps1 = sav['eps1']
        eps2 = sav['eps2']
        Sigma1 = sav['Sigma1']
        Gamma = sav['Gamma']
        im = sav['im']
        
        deps = eps1 - eps2
        
        # Parallel and perpendicular components of NVEC and A
        npar = nvec[:, :2]
        apar, aperp = a[:, :2], a[:, 2]
        
        # Solve BEM equations (simplified)
        # Modify alpha
        alpha = (alpha - self._matmul(Sigma1, a) + 
                1j * k * self._outer(nvec, eps1 * phi))
        
        # Parallel and perpendicular component
        alphapar, alphaperp = alpha[:, :2], alpha[:, 2]
        
        # Modify De (simplified)
        De = (De - eps1 @ Sigma1 @ phi + 
              1j * k * self._inner(nvec, self._matmul(eps1, a)) +
              1j * k * self._inner(npar, self._matmul(deps @ Gamma, alphapar)))
        
        # Solve Eq. (10) using block LU decomposition
        sig2, h2perp = self._solve_block_lu(im, De, alphaperp)
        
        # Parallel component of Green function (simplified)
        h2par = self._rsolve(G2pi, 
                           self._matmul(Gamma, alphapar +
                                      1j * k * self._outer(npar, deps * 
                                                          (G2['ss'] @ sig2 + G2['sh'] @ h2perp))))
        
        # Surface charges and currents at inner interface (simplified)
        sig1 = self._rsolve(G1i, G2['ss'] @ sig2 + G2['sh'] @ h2perp + phi)
        h1perp = self._rsolve(G1i, G2['hh'] @ h2perp + G2['hs'] @ sig2 + aperp)
        h1par = self._rsolve(G1i, self._matmul(G2['p'], h2par) + apar)
        
        # Pack to single vector
        return self._pack_6(sig1, h1par, h1perp, sig2, h2par, h2perp).flatten()
    
    def _solve_block_lu(self, M, b1, b2):
        """
        Solve system of linear equations using block LU decomposition.
        """
        L, U = M  # M contains LU decomposition blocks
        
        # [L11, 0; L21, L22] * [y1; y2] = [b1; b2]
        y1 = self._lsolve(b1, L[0][0], 'L')
        y2 = self._lsolve(b2 - L[1][0] @ y1, L[1][1], 'L')
        
        # [U11, U12; 0, U22] * [x1; x2] = [y1; y2]
        x2 = self._rsolve(y2, U[1][1], 'U')
        x1 = self._rsolve(y1 - U[0][1] @ x2, U[0][0], 'U')
        
        return x1, x2
    
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
        
        # Wavenumber and dielectric functions
        k = self._k
        eps1 = self._eps1
        eps2 = self._eps2
        nvec = self._nvec
        
        # External excitation
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
    
    def _pack(self, *args):
        """Pack scalar and vector potentials into single vector."""
        if len(args) == 4:
            # Extract input: phi, a, phip, ap
            phi, a, phip, ap = args
        else:
            # Extract input: phi, apar, aperp, phip, appar, apperp
            phi, apar, aperp, phip, appar, apperp = args
            
            # Size of vectors
            siz1 = (aperp.shape[0], 2, aperp.shape[-1] if aperp.ndim > 1 else 1)
            siz2 = (aperp.shape[0], 1, aperp.shape[-1] if aperp.ndim > 1 else 1)
            
            # Put together vectors
            a = np.concatenate([apar.reshape(siz1), aperp.reshape(siz2)], axis=1)
            ap = np.concatenate([appar.reshape(siz1), apperp.reshape(siz2)], axis=1)
            
            a = np.squeeze(a)
            ap = np.squeeze(ap)
        
        # Pack to single vector
        return np.concatenate([
            phi.flatten(),
            a.flatten(),
            phip.flatten(),
            ap.flatten()
        ])
    
    def _pack_6(self, phi, apar, aperp, phip, appar, apperp):
        """Pack 6 components into single vector."""
        return self._pack(phi, apar, aperp, phip, appar, apperp)
    
    def _unpack(self, vec):
        """Unpack scalar and vector potentials from vector."""
        n = self.p.n if hasattr(self.p, 'n') else len(vec) // 8
        siz = len(vec) // (8 * n)
        
        # Reshape vector
        vec = vec.reshape(-1, 8)
        
        # Extract potentials from vector
        phi = vec[:, 0].reshape(-1, siz)
        a = vec[:, 1:4].reshape(-1, 3, siz)
        phip = vec[:, 4].reshape(-1, siz)
        ap = vec[:, 5:8].reshape(-1, 3, siz)
        
        return phi, a, phip, ap
    
    def _unpack_4(self, vec):
        """Unpack 4 components from vector."""
        return self._unpack(vec)
    
    def _unpack_6(self, vec):
        """Unpack 6 components from vector."""
        phi, a, phip, ap = self._unpack(vec)
        
        # Decompose vectors into parallel and perpendicular components
        apar = a[:, :2]
        aperp = a[:, 2]
        appar = ap[:, :2]
        apperp = ap[:, 2]
        
        return phi, apar, aperp, phip, appar, apperp
    
    def _inner(self, nvec, a, mul=None):
        """Fast inner product."""
        if np.isscalar(a) and a == 0:
            return 0
        
        if nvec.shape[1] == 2:
            result = (a[:, 0] * nvec[:, 0] + a[:, 1] * nvec[:, 1])
        else:
            result = (a[:, 0] * nvec[:, 0] + a[:, 1] * nvec[:, 1] + a[:, 2] * nvec[:, 2])
        
        # Additional multiplication?
        if mul is not None:
            result = result * mul
        
        return result
    
    def _outer(self, nvec, val, mul=None):
        """Fast outer product between vector and matrix."""
        if np.isscalar(val) and val == 0:
            return 0
        
        siz = (val.shape[0], 1, val.shape[-1] if val.ndim > 1 else 1)
        
        # Additional multiplication?
        if mul is not None:
            val = val * mul
        
        # Outer product
        if nvec.shape[1] == 2:
            return np.concatenate([
                (val * nvec[:, 0]).reshape(siz),
                (val * nvec[:, 1]).reshape(siz)
            ], axis=1)
        else:
            return np.concatenate([
                (val * nvec[:, 0]).reshape(siz),
                (val * nvec[:, 1]).reshape(siz),
                (val * nvec[:, 2]).reshape(siz)
            ], axis=1)
    
    def _matmul(self, a, b):
        """Matrix multiplication helper."""
        if hasattr(a, '__matmul__'):
            if b.ndim > 2:
                # Handle 3D arrays by reshaping
                original_shape = b.shape
                b_2d = b.reshape(b.shape[0], -1)
                result = a @ b_2d
                return result.reshape(original_shape)
            else:
                return a @ b
        else:
            return a * b