"""
BEM solver for quasistatic approximation and eigenmode expansion using
mirror symmetry. Given an external excitation, BEMSTATEIGMIRROR
computes the surface charges such that the boundary conditions of
Maxwell's equations in the quasistatic approximation (using an
eigenmode expansion) are fulfilled.
"""

import numpy as np
from scipy.sparse.linalg import eigs
from ..Base.bembase import BemBase
from ...utils.compstruct import CompStruct
from ...utils.compstructmirror import CompStructMirror, compstructmirror
from ...utils.compgreenstatmirror import compgreenstatmirror
from ...utils.bemoptions import getbemoptions
from ...utils.matrix_operations import matmul


class BemStateigMirror(BemBase):
    """BEM solver for quasistatic approximation with eigenmode expansion using mirror symmetry."""
    
    # Class constants
    name = 'bemsolver'
    needs = [{'sim': 'stat'}, 'nev', 'sym']
    
    def __init__(self, p, *args, **kwargs):
        """
        Initialize quasistatic BEM solver with eigenmodes and mirror symmetry.
        
        Parameters
        ----------
        p : object
            Composite particle with mirror symmetry (see comparticlemirror)
        enei : float, optional
            Light wavelength in vacuum
        op : dict, optional
            Options dictionary
        **kwargs : dict
            Additional options as keyword arguments
        """
        super().__init__()
        self.p = None           # composite particle (see comparticlemirror)
        self.nev = None         # number of eigenmodes
        self.ur = None          # right eigenvectors of surface derivative of Green function
        self.ul = None          # left eigenvectors
        self.ene = None         # eigenvalues
        self.unit = None        # inner product of right and left eigenvectors
        self.enei = None        # light wavelength in vacuum
        
        # Private attributes
        self._g = None          # Green function (needed in field method)
        self._mat = None        # -inv(Lambda + F) computed from eigenmodes
        
        self._init(p, *args, **kwargs)
    
    def __str__(self):
        """String representation."""
        return f"bemstateigmirror: p={self.p}, nev={self.nev}, ur={type(self.ur)}, enei={self.enei}"
    
    def display(self):
        """Command window display."""
        print("bemstateigmirror:")
        print({
            'p': self.p,
            'nev': self.nev,
            'ur': self.ur,
            'enei': self.enei
        })
    
    def _init(self, p, *args, **kwargs):
        """
        Initialize quasistatic BEM solver with eigenmode expansion.
        
        Parameters
        ----------
        p : object
            Composite particle
        enei : float, optional
            Light wavelength in vacuum (if first arg is numeric)
        **kwargs : dict
            Additional options
        """
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
        op = getbemoptions(**options)
        
        # Save particle and number of eigenvalues
        self.p = p
        self.nev = op.nev
        
        # Green function
        self._g = compgreenstatmirror(p, p, op)
        
        # Surface derivative of Green function
        F = self._g.F
        
        # Eigenmode expansion
        opts = {'maxiter': 1000}
        
        # Initialize lists for symmetry values
        self.ur = []
        self.ul = []
        self.ene = []
        self.unit = []
        
        # Loop over symmetry values
        for i in range(len(F)):
            # Plasmon eigenmodes (left and right eigenvectors)
            ul, _ = eigs(F[i].T, self.nev, which='SR', **opts)
            ul = ul.T
            ur, ene = eigs(F[i], self.nev, which='SR', **opts)
            
            # Make eigenvectors orthogonal (needed for degenerate eigenvalues)
            ul = np.linalg.solve(ul @ ur, ul)
            
            # Unit matrices
            unit = np.zeros((self.nev**2, p.np))
            
            # Loop over unique material combinations
            for ip in range(p.np):
                ind = p.index[ip]
                unit[:, ip] = (ul[:, ind] @ ur[ind, :]).reshape(self.nev**2)
            
            # Save eigenmodes in BEM solver
            self.ur.append(ur)
            self.ul.append(ul)
            self.ene.append(ene)
            self.unit.append(unit)
        
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
            # Dielectric function
            eps = np.array([eps_func(enei) for eps_func in self.p.eps])
            
            # Inside and outside dielectric function
            eps1 = eps[self.p.inout[:, 0]]
            eps2 = eps[self.p.inout[:, 1]]
            
            # Lambda [Garcia de Abajo, Eq. (23)]
            Lambda = 2 * np.pi * (eps1 + eps2) / (eps1 - eps2)
            
            # Initialize matrix list
            self._mat = []
            
            for i in range(len(self.ur)):
                # BEM resolvent matrix
                Lambda_reshaped = (self.unit[i] @ Lambda.flatten()).reshape(self.nev, -1)
                mat_i = -self.ur[i] @ np.linalg.solve(Lambda_reshaped + self.ene[i], self.ul[i])
                self._mat.append(mat_i)
            
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
        self : BemStateigMirror
            Self for method chaining
        """
        self._initialize_matrices(enei)
        return self
    
    def solve(self, exc):
        """
        Surface charge for given excitation (mldivide equivalent).
        
        Parameters
        ----------
        exc : CompStructMirror
            CompStructMirror with field 'phip' for external excitation
            
        Returns
        -------
        sig : CompStructMirror
            CompStructMirror with field for surface charge
        """
        # Initialize BEM solver (if needed)
        self._initialize_matrices(exc.enei)
        
        # Initialize surface charges
        sig = compstructmirror(self.p, exc.enei, exc.fun)
        
        # Loop over excitations
        for i in range(len(exc.val)):
            # Get symmetry value
            ind = self.p.symindex(exc.val[i].symval[-1, :])
            
            # Surface charge
            sig_values = matmul(self._mat[ind], exc.val[i].phip)
            sig.val[i] = CompStruct(self.p, exc.enei, sig=sig_values)
            
            # Set symmetry value
            sig.val[i].symval = exc.val[i].symval
        
        return sig
    
    def __truediv__(self, exc):
        """Operator overload for solve (\ in MATLAB)."""
        return self.solve(exc)
    
    def induced_potential(self, sig):
        """
        Induced potential for given surface charge (mtimes equivalent).
        
        Parameters
        ----------
        sig : CompStructMirror
            CompStructMirror with fields for surface charge
            
        Returns
        -------
        phi : CompStructMirror
            CompStructMirror with fields for induced potential
        """
        phi = self.potential(sig, 1)
        phi2 = self.potential(sig, 2)
        
        for i in range(len(sig)):
            phi[i] = phi[i] + phi2[i]
        
        return phi
    
    def __mul__(self, sig):
        """Operator overload for induced_potential (* in MATLAB)."""
        return self.induced_potential(sig)
    
    def field(self, sig, inout=2):
        """
        Electromagnetic fields inside or outside of particle surface.
        
        Computed from solutions of Maxwell equations within the
        quasistatic approximation.
        
        Parameters
        ----------
        sig : CompStructMirror
            CompStructMirror object with surface charges
        inout : int, optional
            Electric field inside (inout=1) or outside (inout=2, default)
            
        Returns
        -------
        field : CompStructMirror
            CompStructMirror object with electric field
        """
        # Field from Green function
        return self._g.field(sig, inout)
    
    def potential(self, sig, inout=2):
        """
        Potentials and surface derivatives inside or outside of particle.
        
        Computed from solutions of Maxwell equations within the
        quasistatic approximation.
        
        Parameters
        ----------
        sig : CompStructMirror
            CompStructMirror with surface charges
        inout : int, optional
            Potentials inside (inout=1) or outside (inout=2, default)
            
        Returns
        -------
        pot : CompStructMirror
            CompStructMirror object with potentials
        """
        # Field from Green function
        return self._g.potential(sig, inout)


def bemstateigmirror(p, *args, **kwargs):
    """
    Factory function to create BemStateigMirror instance.
    
    Parameters
    ----------
    p : object
        Composite particle with mirror symmetry
    *args : tuple
        Additional positional arguments
    **kwargs : dict
        Additional keyword arguments
        
    Returns
    -------
    BemStateigMirror
        Initialized BEM solver instance
    """
    return BemStateigMirror(p, *args, **kwargs)