"""
Iterative BEM Retarded Solver - Python Implementation

Iterative BEM solver for full Maxwell equations.
Given an external excitation, BEMRETITER computes the surface charges 
iteratively such that the boundary conditions of Maxwell's equations are fulfilled.

See, e.g. Garcia de Abajo and Howie, PRB 65, 115418 (2002).
Translated from MATLAB MNPBEM toolbox @bemretiter class.
"""

import numpy as np
from typing import Optional, Tuple, Any, Dict, Union, Callable
from ..Base.bembase import BemBase
from .bemiter import BemIter


class BemRetIter(BemBase, BemIter):
    """
    Iterative BEM solver for full Maxwell equations.
    
    Given an external excitation, BEMRETITER computes the surface
    charges iteratively such that the boundary conditions of Maxwell's
    equations are fulfilled.
    
    Reference: Garcia de Abajo and Howie, PRB 65, 115418 (2002).
    """
    
    # BemBase abstract properties
    name = 'bemsolver'
    needs = [{'sim': 'ret'}, 'iter']
    
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
        # Initialize BemIter first (for iterative solver properties)
        if len(args) >= 2:
            BemIter.__init__(self, *args[1:], **kwargs)
        else:
            BemIter.__init__(self, **kwargs)
        
        # Public properties specific to bemretiter
        self.p = None           # compound of discretized particles
        self.g = None           # Green function
        self.enei = []          # light wavelength in vacuum
        
        # Private properties specific to bemretiter
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
        return (f"BemRetIter(p={type(self.p).__name__ if self.p else None}, "
                f"g={type(self.g).__name__ if self.g else None}, "
                f"solver='{self.solver}', tol={self.tol})")
    
    def __repr__(self):
        """Command window display."""
        return (f"bemretiter:\n{{'p': {self.p}, 'g': {self.g}, "
                f"'solver': '{self.solver}', 'tol': {self.tol}}}")
    
    def _init(self, p, *args, **kwargs):
        """
        Initialize iterative BEM solver.
        
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
        self._nvec = p.nvec if hasattr(p, 'nvec') else np.eye(3)[:100]  # Placeholder
        
        # Handle calls with and without ENEI
        varargin = list(args)
        enei = None
        if varargin and isinstance(varargin[0], (int, float, complex)):
            enei = varargin[0]
            varargin = varargin[1:]
        
        # Option structure
        self._op = self._getbemoptions(['iter', 'bemiter'], *varargin, **kwargs)
        
        # Green function, set tolerance and maximal rank for low-rank matrices
        htol = min(self._op.get('htol', [1e-6])) if isinstance(self._op.get('htol'), list) else self._op.get('htol', 1e-6)
        kmax = max(self._op.get('kmax', [100])) if isinstance(self._op.get('kmax'), list) else self._op.get('kmax', 100)
        
        self.g = self._compgreenret_aca(p, htol=htol, kmax=kmax, **kwargs)
        
        # Cluster tree for reduction of preconditioner matrices
        if 'reduce' in self._op:
            self.rtree = self._reducetree(self.g.hmat.tree, self._op['reduce'])
        
        # Initialize for given wavelength
        if enei is not None:
            self._initmat(enei)
    
    def _compgreenret_aca(self, p, **options):
        """
        Placeholder for ACA Green function initialization.
        In actual implementation, this would create the ACA retarded Green function.
        """
        # This would be implemented with actual ACA Green function computation
        return type('ACAGreenFunction', (), {
            'G': lambda enei: np.eye(100),  # Placeholder
            'H1': lambda enei: np.eye(100),  # Placeholder
            'H2': lambda enei: np.eye(100),  # Placeholder
            'potential': lambda sig, inout: sig,  # Placeholder
            'field': lambda sig, inout: sig,  # Placeholder
            'deriv': 'norm',
            'hmat': type('HMat', (), {'tree': None})()
        })()
    
    def _reducetree(self, tree, reduce_param):
        """Placeholder for tree reduction."""
        return tree
    
    def clear(self):
        """
        Clear Green functions and auxiliary matrices.
        
        Returns:
        --------
        obj : BemRetIter
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
        
        Parameters:
        -----------
        enei : float
            Light wavelength in vacuum
        """
        # Use previously computed matrices?
        if not self.enei or self.enei != enei:
            self.enei = enei
            
            # Wavenumber
            self._k = 2 * np.pi / enei
            
            # Dielectric function
            self._eps1 = self.p.eps1(enei) if hasattr(self.p, 'eps1') else 1.0
            self._eps2 = self.p.eps2(enei) if hasattr(self.p, 'eps2') else 1.0
            
            self.tocout('init', 'G1', 'G2', 'H1', 'H2', 'G1i', 'G2i',
                       'Sigma1', 'Sigma2', 'Deltai', 'Sigmai')
            
            # Green functions and surface derivatives (placeholder)
            self._G1 = self.g.G(enei) - self.g.G(enei)  # Placeholder
            self.tocout('G1')
            self._G2 = self.g.G(enei) - self.g.G(enei)  # Placeholder
            self.tocout('G2')
            
            self._H1 = self.g.H1(enei) - self.g.H1(enei)  # Placeholder
            self.tocout('H1')
            self._H2 = self.g.H2(enei) - self.g.H2(enei)  # Placeholder
            self.tocout('H2')
            
            # Initialize preconditioner
            if self.precond:
                self._initprecond(enei)
            
            self.tocout('close')
            
            # Save statistics
            self.setstat('G1', self._G1)
            self.setstat('H1', self._H1)
            self.setstat('G2', self._G2)
            self.setstat('H2', self._H2)
    
    def _initprecond(self, enei: float):
        """
        Initialize preconditioner for H-matrices.
        
        Reference: Garcia de Abajo and Howie, PRB 65, 115418 (2002).
        """
        # Wavenumber
        k = 2 * np.pi / enei
        
        # Dielectric functions
        eps1 = self._spdiag(self._eps1)
        eps2 = self._spdiag(self._eps2)
        
        # Normal vector
        nvec = self._nvec
        
        # Green functions and surface derivatives
        G1 = self._compress(self._G1)
        H1 = self._compress(self._H1)
        G2 = self._compress(self._G2)
        H2 = self._compress(self._H2)
        
        # Use hierarchical or full matrices for preconditioner?
        if self.precond == 'hmat':
            inv2 = lambda x: self._lu_decomp(x)
            mul2 = lambda x, y: self._rsolve(x, y)
        elif self.precond == 'full':
            # Expand matrices
            G1, H1 = self._to_full(G1), self._to_full(H1)
            G2, H2 = self._to_full(G2), self._to_full(H2)
            # Inversion and multiplication functions
            inv2 = lambda x: np.linalg.inv(x)
            mul2 = lambda x, y: x @ y
        else:
            return
        
        # Inverse Green function
        G1i = inv2(G1)
        self.tocout('G1i')
        G2i = inv2(G2)
        self.tocout('G2i')
        
        # Sigma matrices [Eq.(21)]
        Sigma1 = mul2(H1, G1i)
        self.tocout('Sigma1')
        Sigma2 = mul2(H2, G2i)
        self.tocout('Sigma2')
        
        # Inverse Delta matrix
        Deltai = np.linalg.inv(Sigma1 - Sigma2)
        self.tocout('Deltai')
        
        deps = eps1 - eps2
        
        # Sigma matrix
        Sigma = (eps1 * Sigma1 - eps2 * Sigma2 + 
                k**2 * deps * self._fun_deltai(Deltai, nvec) * deps)
        
        # Save variables
        sav = {
            'k': k,
            'nvec': nvec,
            'G1i': G1i,
            'G2i': G2i,
            'eps1': eps1,
            'eps2': eps2,
            'Sigma1': Sigma1,
            'Deltai': Deltai,
            'Sigmai': inv2(Sigma)
        }
        
        # Save structure
        self._sav = sav
        
        # Save statistics
        if self.precond == 'hmat':
            hmat_list = [G1i, G2i, Sigma1, Sigma2, Deltai, sav['Sigmai']]
            name_list = ['G1i', 'G2i', 'Sigma1', 'Sigma2', 'Deltai', 'Sigmai']
            
            for hmat, name in zip(hmat_list, name_list):
                self.setstat(name, hmat)
    
    def _fun_deltai(self, Deltai, nvec):
        """
        Decorate Deltai function with normal vectors, Eq. (21).
        """
        nvec1 = self._spdiag(nvec[:, 0])
        nvec2 = self._spdiag(nvec[:, 1])
        nvec3 = self._spdiag(nvec[:, 2])
        
        return (nvec1 @ Deltai @ nvec1 + 
                nvec2 @ Deltai @ nvec2 + 
                nvec3 @ Deltai @ nvec3)
    
    def _compress(self, hmat):
        """
        Compress H-matrices for preconditioner.
        """
        # Change tolerance and maximal rank for low-rank matrices
        if hasattr(hmat, 'htol'):
            hmat.htol = max(self._op.get('htol', [1e-6]))
        if hasattr(hmat, 'kmax'):
            hmat.kmax = min(self._op.get('kmax', [100]))
        return hmat
    
    def _lu_decomp(self, matrix):
        """LU decomposition placeholder."""
        return matrix  # Would implement actual LU decomposition
    
    def _rsolve(self, lu_matrix, vector):
        """Resolve with LU decomposition placeholder."""
        return vector  # Would implement actual resolution
    
    def _to_full(self, matrix):
        """Convert to full matrix."""
        if hasattr(matrix, 'toarray'):
            return matrix.toarray()
        return matrix
    
    def __call__(self, enei: float):
        """
        Initialize matrices for given energy.
        
        Parameters:
        -----------
        enei : float
            Light wavelength in vacuum
            
        Returns:
        --------
        obj : BemRetIter
            Updated object with initialized matrices
        """
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
        obj : BemRetIter
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
    
    def __mul__(self, sig):
        """Induced potential for given surface charge."""
        return self.potential(sig, 1) + self.potential(sig, 2)
    
    def potential(self, sig, inout: int = 2):
        """
        Determine potentials and surface derivatives inside/outside of particle.
        """
        return self.g.potential(sig, inout)
    
    def field(self, sig, inout: int = 2):
        """
        Electric and magnetic field inside/outside of particle surface.
        """
        # Simplified implementation - placeholder for actual field calculation
        k = 2 * np.pi / sig.enei
        pot = self.potential(sig, inout)
        
        # Extract fields (simplified)
        if hasattr(pot, 'phi1'):
            phi, phip, a, ap = pot.phi1, pot.phi1p, pot.a1, pot.a1p
        else:
            phi, phip, a, ap = pot.phi2, pot.phi2p, pot.a2, pot.a2p
        
        # Simplified field calculation
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
        
        Reference: Garcia de Abajo and Howie, PRB 65, 115418 (2002).
        """
        n = self.p.n if hasattr(self.p, 'n') else len(vec) // 8
        siz = len(vec) // 2
        
        # Split vector array
        vec1 = vec[:siz].reshape(n, -1)
        vec2 = vec[siz:].reshape(n, -1)
        
        # Multiplication with Green functions
        G_result = np.concatenate([
            (self._G1 @ vec1).reshape(-1),
            (self._G2 @ vec2).reshape(-1)
        ])
        
        H_result = np.concatenate([
            (self._H1 @ vec1).reshape(-1),
            (self._H2 @ vec2).reshape(-1)
        ])
        
        Gsig1, Gh1, Gsig2, Gh2 = self._unpack(G_result)
        Hsig1, Hh1, Hsig2, Hh2 = self._unpack(H_result)
        
        # Extract input
        k, nvec, eps1, eps2 = self._k, self._nvec, self._eps1, self._eps2
        
        # Eq. (10)
        phi = Gsig1 - Gsig2
        
        # Eq. (11)
        a = Gh1 - Gh2
        
        # Eq. (14)
        alpha = (Hh1 - Hh2 - 
                1j * k * self._outer(nvec, self._matmul(eps1, Gsig1) - 
                                     self._matmul(eps2, Gsig2)))
        
        # Eq. (17)
        De = (self._matmul(eps1, Hsig1) - self._matmul(eps2, Hsig2) -
              1j * k * self._inner(nvec, self._matmul(eps1, Gh1) - 
                                   self._matmul(eps2, Gh2)))
        
        # Pack into single vector
        return self._pack(phi, a, De, alpha)
    
    def _mfun(self, vec):
        """
        Preconditioner for CGS, BICGSTAB, GMRES.
        
        Reference: Garcia de Abajo and Howie, PRB 65, 115418 (2002).
        """
        # Unpack matrices
        phi, a, De, alpha = self._unpack(vec)
        
        # Get variables for evaluation of preconditioner
        sav = self._sav
        k = sav['k']
        nvec = sav['nvec']
        G1i = sav['G1i']
        G2i = sav['G2i']
        eps1 = sav['eps1']
        eps2 = sav['eps2']
        Sigma1 = sav['Sigma1']
        Deltai = sav['Deltai']
        Sigmai = sav['Sigmai']
        
        # Define matrix multiplication functions
        if self.precond == 'hmat':
            matmul1 = lambda a, b: (a @ b.reshape(b.shape[0], -1)).reshape(b.shape)
            matmul2 = lambda a, b: (self._rsolve(a, b.reshape(b.shape[0], -1))).reshape(b.shape)
        else:
            matmul1 = lambda a, b: (a @ b.reshape(b.shape[0], -1)).reshape(b.shape)
            matmul2 = lambda a, b: (a @ b.reshape(b.shape[0], -1)).reshape(b.shape)
        
        # Modify alpha and De
        alpha = (alpha - matmul1(Sigma1, a) + 
                1j * k * self._outer(nvec, eps1 * phi))
        De = (De - eps1 * matmul1(Sigma1, phi) + 
              1j * k * eps1 * self._inner(nvec, a))
        
        # Eq. (19)
        sig2 = matmul2(Sigmai, 
                      De + 1j * k * (eps1 - eps2) * 
                      self._inner(nvec, matmul1(Deltai, alpha)))
        
        # Eq. (20)
        h2 = matmul1(Deltai, 
                    1j * k * self._outer(nvec, (eps1 - eps2) * sig2) + alpha)
        
        # Surface charges and currents
        sig1 = matmul2(G1i, sig2 + phi)
        h1 = matmul2(G1i, h2 + a)
        sig2 = matmul2(G2i, sig2)
        h2 = matmul2(G2i, h2)
        
        # Pack to vector
        return self._pack(sig1, h1, sig2, h2).flatten()
    
    def _excitation(self, exc):
        """Compute excitation variables for iterative BEM solver."""
        # Default values for potentials
        phi1, phi1p, a1, a1p = 0, 0, 0, 0
        phi2, phi2p, a2, a2p = 0, 0, 0, 0
        
        # Extract fields (simplified)
        if hasattr(exc, 'phi1'):
            phi1 = exc.phi1
        if hasattr(exc, 'phi2'):
            phi2 = exc.phi2
        # ... similar for other fields
        
        # Wavenumber of light in vacuum
        k = 2 * np.pi / exc.enei
        
        # Dielectric functions
        eps1 = self._eps1
        eps2 = self._eps2
        
        # Outer surface normal
        nvec = self._nvec
        
        # External excitation - Garcia de Abajo and Howie, PRB 65, 115418 (2002)
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
        
        # Expand arrays
        if np.isscalar(phi) and phi == 0:
            phi = np.zeros_like(De)
        if np.isscalar(a) and a == 0:
            a = np.zeros_like(alpha)
        
        return phi, a, De, alpha
    
    def _pack(self, phi, a, phip, ap):
        """Pack scalar and vector potentials into single vector."""
        return np.concatenate([
            phi.flatten(),
            a.flatten(),
            phip.flatten(),
            ap.flatten()
        ])
    
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
    
    def _inner(self, nvec, a, mul=None):
        """Fast inner product."""
        if np.isscalar(a) and a == 0:
            return 0
        
        # Inner product
        result = (a[:, 0] * nvec[:, 0] + 
                 a[:, 1] * nvec[:, 1] + 
                 a[:, 2] * nvec[:, 2])
        
        # Additional multiplication?
        if mul is not None:
            result = result * mul
        
        return result
    
    def _outer(self, nvec, val, mul=None):
        """Fast outer product between vector and matrix."""
        if np.isscalar(val) and val == 0:
            return 0
        
        # Additional multiplication?
        if mul is not None:
            val = val * mul
        
        # Outer product
        return np.stack([
            val * nvec[:, 0],
            val * nvec[:, 1],
            val * nvec[:, 2]
        ], axis=1)
    
    def _matmul(self, a, b):
        """Fast matrix multiplication."""
        if np.isscalar(b) and b == 0:
            return 0
        if hasattr(a, '__matmul__'):
            return a @ b
        else:
            return a * b
    
    def _spdiag(self, values):
        """Create sparse diagonal matrix or return scalar."""
        if hasattr(values, '__len__') and len(values) > 1:
            return np.diag(values)
        else:
            return values