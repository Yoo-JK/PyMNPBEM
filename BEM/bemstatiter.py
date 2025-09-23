"""
Iterative BEM solver for quasistatic approximation.

Given an external excitation, BEMSTATITER computes the surface
charges such that the boundary conditions of Maxwell's equations
in the quasistatic approximation are fulfilled. Maxwell's equations
are solved iteratively.

See, e.g. Garcia de Abajo and Howie, PRB 65, 115418 (2002);
                Hohenester et al., PRL 103, 106801 (2009).
"""

import numpy as np
from scipy.sparse import diags as spdiag
from scipy.linalg import inv
from ..Base.bembase import BemBase
from .bemiter import BemIter
from ...utils.compstruct import CompStruct
from ...utils.bemoptions import getbemoptions
from ...utils.matrix_operations import matmul, outer
from ...utils import aca


class BemStatIter(BemBase, BemIter):
    """Iterative BEM solver for quasistatic approximation."""
    
    # Class constants
    name = 'bemsolver'
    needs = [{'sim': 'stat'}, 'iter']
    
    def __init__(self, p, *args, **kwargs):
        """
        Initialize quasistatic, iterative BEM solver.
        
        Parameters
        ----------
        p : object
            Composite particle (see comparticle)
        enei : float, optional
            Light wavelength in vacuum
        op : dict, optional
            Options dictionary
        **kwargs : dict
            Additional options as keyword arguments
        """
        # Initialize parent classes
        BemIter.__init__(self, *args[1:], **kwargs)
        BemBase.__init__(self)
        
        self.p = None           # composite particle
        self.F = None           # surface derivative of Green function
        self.enei = None        # light wavelength in vacuum
        
        # Private attributes
        self._op = None         # option structure
        self._g = None          # Green function object
        self._lambda = None     # resolvent matrix is -inv(lambda + F)
        self._mat = None        # -inv(Lambda + F) computed with H-matrix inversion
        
        self._init(p, *args, **kwargs)
    
    def __str__(self):
        """String representation."""
        return f"bemstatiter: p={self.p}, F={type(self.F)}, solver={self.solver}, tol={self.tol}"
    
    def clear(self):
        """Clear auxiliary matrices."""
        self._mat = None
        return self
    
    def _init(self, p, *args, **kwargs):
        """
        Initialize quasistatic, iterative BEM solver.
        
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
        
        # Get options
        self._op = getbemoptions(['iter', 'bemiter'], **options)
        
        # Green function
        htol = min(self._op.htol) if hasattr(self._op, 'htol') else 1e-3
        kmax = max(self._op.kmax) if hasattr(self._op, 'kmax') else 200
        self._g = aca.compgreenstat(p, htol=htol, kmax=kmax, **options)
        
        # Surface derivative of Green function
        self.F = self._g.F
        self.F.val = self.F.val.reshape(-1, 1)
        
        # Add statistics
        self.setstat('F', self.F)
        
        # Initialize for given wavelength
        if enei is not None:
            self._initmat(enei)
    
    def _initmat(self, enei):
        """
        Initialize Green functions and preconditioner for iterative BEM solver.
        
        Parameters
        ----------
        enei : float
            Light wavelength in vacuum
        """
        # Use previously computed matrices?
        if self.enei is None or self.enei != enei:
            self.enei = enei
            
            # Dielectric functions
            eps1 = self.p.eps1(enei)
            eps2 = self.p.eps2(enei)
            
            # Lambda function [Garcia de Abajo, Eq. (23)]
            self._lambda = 2 * np.pi * (eps1 + eps2) / (eps1 - eps2)
            
            # Initialize preconditioner
            if hasattr(self, 'precond') and self.precond:
                # Surface derivative of Green function
                F = self.F
                lambda_diag = spdiag(self._lambda)
                
                # Resolvent matrix
                if self.precond == 'hmat':
                    # Change tolerance and maximum rank
                    F.htol = max(self._op.htol) if hasattr(self._op, 'htol') else 1e-2
                    F.kmax = min(self._op.kmax) if hasattr(self._op, 'kmax') else 100
                    
                    # Initialize preconditioner
                    self._mat = (-lambda_diag - F).lu()
                    
                    # Save statistics for H-matrix operation
                    self.setstat('mat', self._mat)
                    
                elif self.precond == 'full':
                    # Initialize preconditioner
                    self._mat = inv(-lambda_diag - F.full())
                    
                else:
                    raise ValueError('preconditioner not known')
    
    def __call__(self, enei):
        """
        Initialize BEM solver for given energy/wavelength.
        
        Parameters
        ----------
        enei : float
            Light wavelength in vacuum
            
        Returns
        -------
        self : BemStatIter
            Self for method chaining
        """
        self._initmat(enei)
        return self
    
    def _afun(self, vec):
        """
        Matrix multiplication for iterative solvers.
        
        Parameters
        ----------
        vec : ndarray
            Input vector
            
        Returns
        -------
        ndarray
            Result of -(lambda + F) * vec
        """
        # Unpack vector
        vec = vec.reshape(self.p.n, -1)
        
        # -(lambda + F) * vec
        vec = -(self.F @ vec + vec * self._lambda[:, np.newaxis])
        
        return vec.reshape(-1, 1)
    
    def _mfun(self, vec):
        """
        Preconditioner for iterative solvers.
        
        Parameters
        ----------
        vec : ndarray
            Input vector
            
        Returns
        -------
        ndarray
            Preconditioned vector
        """
        vec = vec.reshape(self.p.n, -1)
        
        # Preconditioner
        if self.precond == 'hmat':
            vec = self._mat.solve(vec)
        elif self.precond == 'full':
            vec = self._mat @ vec
        
        return vec.flatten()
    
    def solve(self, exc):
        """
        Solve BEM equations for given excitation.
        
        Parameters
        ----------
        exc : CompStruct
            CompStruct with fields for external excitation
            
        Returns
        -------
        sig : CompStruct
            CompStruct with fields for surface charge
        """
        # Initialize BEM solver (if needed)
        self._initmat(exc.enei)
        
        # Excitation and size of excitation array
        b = exc.phip.flatten()
        siz = exc.phip.shape
        
        # Function for matrix multiplication
        def fa(x):
            return self._afun(x)
        
        # Function for preconditioner
        fm = None
        if hasattr(self, 'precond') and self.precond:
            def fm(x):
                return self._mfun(x)
        
        # Iterative solution
        x = self.solve_iterative(None, b, fa, fm)
        
        # Save everything in single structure
        sig = CompStruct(self.p, exc.enei, sig=x.reshape(siz))
        
        return sig
    
    def __truediv__(self, exc):
        """Surface charge for given excitation (mldivide equivalent)."""
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
        # Electric field in normal direction
        if inout == 1:
            e = -outer(self.p.nvec, matmul(self._g.H1, sig.sig))
        else:  # inout == 2
            e = -outer(self.p.nvec, matmul(self._g.H2, sig.sig))
        
        # Tangential directions of electric field are computed by
        # interpolation of potential to vertices and performing an
        # approximate tangential derivative
        phi = self.p.interp(matmul(self._g.G, sig.sig))
        
        # Derivatives of function along tangential directions
        phi1, phi2, t1, t2 = self.p.deriv(phi)
        
        # Normal vector
        nvec = np.cross(t1, t2, axis=1)
        
        # Decompose into norm and unit vector
        h = np.sqrt(np.sum(nvec * nvec, axis=1, keepdims=True))
        nvec = nvec / h
        
        # Tangential derivative of PHI
        phip = (outer(np.cross(t2, nvec, axis=1) / h, phi1) -
               outer(np.cross(t1, nvec, axis=1) / h, phi2))
        
        # Add electric field in tangential direction
        e = e - phip
        
        # Set output
        return CompStruct(self.p, sig.enei, e=e)
    
    def potential(self, sig, inout=2):
        """
        Determine potentials and surface derivatives inside/outside of particle.
        
        Computed from solutions of Maxwell equations within the
        quasistatic approximation.
        
        Parameters
        ----------
        sig : CompStruct
            CompStruct with surface charges
        inout : int, optional
            Potential inside (inout=1) or outside (inout=2, default)
            
        Returns
        -------
        pot : CompStruct
            CompStruct object with potentials
        """
        return self._g.potential(sig, inout)


def bemstatiter(p, *args, **kwargs):
    """
    Factory function to create BemStatIter instance.
    
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
    BemStatIter
        Initialized BEM solver instance
    """
    return BemStatIter(p, *args, **kwargs)