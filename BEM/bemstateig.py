"""
BEM solver for quasistatic approximation and eigenmode expansion.

Given an external excitation, BEMSTATEIG computes the surface
charges such that the boundary conditions of Maxwell's equations
in the quasistatic approximation (using an eigenmode expansion) 
are fulfilled.

See, e.g. Garcia de Abajo and Howie, PRB 65, 115418 (2002);
                Hohenester et al., PRL 103, 106801 (2009).
"""

import numpy as np
from scipy.sparse.linalg import eigs
from ..Base.bembase import BemBase
from ...utils.compstruct import CompStruct
from ...utils.compgreenstat import compgreenstat
from ...utils.bemoptions import getbemoptions
from ...utils.matrix_operations import matmul, outer


class BemStateig(BemBase):
    """BEM solver for quasistatic approximation with eigenmode expansion."""
    
    # Class constants
    name = 'bemsolver'
    needs = [{'sim': 'stat'}, 'nev']
    
    def __init__(self, p, *args, **kwargs):
        """
        Initialize quasistatic BEM solver with eigenmode expansion.
        
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
        super().__init__()
        self.p = None           # composite particle
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
        return f"bemstateig: p={self.p}, nev={self.nev}, ur={type(self.ur)}, enei={self.enei}"
    
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
        
        # Green function
        self._g = compgreenstat(p, p, **options)
        
        # Surface derivative of Green function
        F = self._g.F
        
        # Options for BEM solver
        op = getbemoptions(['bemstateig'], **options)
        
        # Number of eigenvalues
        self.nev = op.get('nev', 40)
        
        # Eigenmode expansion
        opts = {'maxiter': 1000}
        
        # Plasmon eigenmodes (left and right eigenvectors)
        ul, _ = eigs(F.T, self.nev, which='SR', **opts)
        ul = ul.T
        ur, ene = eigs(F, self.nev, which='SR', **opts)
        
        # Make eigenvectors orthogonal (needed for degenerate eigenvalues)
        ul = np.linalg.solve(ul @ ur, ul)
        
        # Unit matrices
        unit = np.zeros((self.nev**2, p.np))
        
        # Loop over unique material combinations
        for i in range(p.np):
            ind = p.index[i]
            unit[:, i] = (ul[:, ind] @ ur[ind, :]).reshape(self.nev**2)
        
        # Save eigenmodes in BEM solver
        self.ur = ur
        self.ul = ul
        self.ene = ene
        self.unit = unit
        
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
            
            # BEM resolvent matrix
            Lambda_reshaped = (self.unit @ Lambda.flatten()).reshape(self.nev, -1)
            self._mat = -self.ur @ np.linalg.solve(Lambda_reshaped + self.ene, self.ul)
            
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
        self : BemStateig
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
            CompStruct with field containing surface charges
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
        phi = self.potential(sig, 1)
        phi2 = self.potential(sig, 2)
        phi.phi2p = phi2.phi2p
        return phi
    
    def __mul__(self, sig):
        """Operator overload for induced_potential (* in MATLAB)."""
        return self.induced_potential(sig)
    
    def field(self, sig, inout=2):
        """
        Electric field inside/outside of particle surface.
        
        Parameters
        ----------
        sig : CompStruct
            CompStruct with surface charges
        inout : int, optional
            Electric field inside (inout=1) or outside (inout=2, default)
            
        Returns
        -------
        field : CompStruct
            CompStruct object with electric field
        """
        # Compute field from derivative of Green function or from potential interpolation
        if self._g.deriv == 'cart':
            return self._g.field(sig, inout)
        
        elif self._g.deriv == 'norm':
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
    
    def fieldstat(self, sig, inout=2):
        """Alias for field method."""
        return self.field(sig, inout)
    
    def potential(self, sig, inout=2):
        """
        Determine potentials and surface derivatives inside/outside of particle.
        
        Computed from solutions of Maxwell equations within the
        quasistatic approximation using an eigenmode expansion.
        
        Parameters
        ----------
        sig : CompStruct
            CompStruct with surface charges
        inout : int, optional
            Inside (inout=1) or outside (inout=2, default) of particle
            
        Returns
        -------
        pot : CompStruct
            CompStruct object with potentials
        """
        return self._g.potential(sig, inout)
    
    def potentialstat(self, sig, inout=2):
        """Alias for potential method."""
        return self.potential(sig, inout)


def bemstateig(p, *args, **kwargs):
    """
    Factory function to create BemStateig instance.
    
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
    BemStateig
        Initialized BEM solver instance
    """
    return BemStateig(p, *args, **kwargs)