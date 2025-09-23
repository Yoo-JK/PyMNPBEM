import numpy as np
from .bembase import BemBase
from .greenret import GreenRet
from .greenstat import GreenStat
from .compstruct import CompStruct
from .blockmatrix import BlockMatrix
from .clustertree import ClusterTree
from .hmatrix import HMatrix
from .slicer import Slicer
from .bemoptions import get_bemoptions, BemOptions


class CompGreenRet(BemBase):
    """Green function for composite points and particle."""
    
    # Class attributes
    name = 'greenfunction'
    needs = [{'sim': 'ret'}]
    
    def __init__(self, *args, **kwargs):
        """
        Initialize Green functions for composite objects.
        
        Parameters
        ----------
        p1 : object
            Green function between points p1 and comparticle p2
        p2 : object
            Green function between points p1 and comparticle p2
        *args, **kwargs : optional
            Options (see BemOptions)
        """
        super().__init__()
        self.p1 = None
        self.p2 = None
        self.con = None
        self.g = None
        self.hmode = None
        self.block = None
        self.hmat = None
        self._init(*args, **kwargs)
    
    def __str__(self):
        """Command window display."""
        info = {
            'p1': self.p1,
            'p2': self.p2,
            'con': self.con,
            'g': self.g,
            'hmode': self.hmode
        }
        return f"CompGreenRet:\n{info}"
    
    def _init(self, p1, p2, *args, **kwargs):
        """Initialize composite Green function."""
        # Save particles and points
        self.p1, self.p2 = p1, p2
        
        # Initialize Green function
        g = GreenRet(p1, p2, *args, **kwargs)
        
        # Deal with closed argument
        g = self._init_closed(g, p1, p2, *args, **kwargs)
        
        # Split Green function
        self.g = self._mat2cell(g, p1.p, p2.p)
        
        # Connectivity matrix
        self.con = self._connect(p1, p2)
        
        # Size of point or particle objects
        siz1 = [p.n for p in p1.p]
        siz2 = [p.n for p in p2.p]
        
        # Block matrix for evaluation of selected Green function elements
        self.block = BlockMatrix(siz1, siz2)
        
        op = get_bemoptions(['green', 'greenret'], *args, **kwargs)
        
        # Hierarchical matrices?
        if hasattr(op, 'hmode') and op.hmode is not None:
            # Mode for hierarchical matrices: 'aca1', 'aca2', 'svd'
            self.hmode = op.hmode
            
            # Set up cluster trees
            tree1 = ClusterTree(p1, op)
            tree2 = ClusterTree(p2, op)
            
            # Initialize hierarchical matrix
            self.hmat = HMatrix(tree1, tree2, op)
    
    def _mat2cell(self, g, p1_parts, p2_parts):
        """Split Green function into cell array."""
        # This would split the Green function matrix into blocks
        # corresponding to different particle parts
        result = {}
        for i, p1_part in enumerate(p1_parts):
            for j, p2_part in enumerate(p2_parts):
                result[(i, j)] = g  # Simplified - would need actual splitting logic
        return result
    
    def _connect(self, p1, p2):
        """Create connectivity matrix."""
        # This would determine which particle parts are connected
        n1 = len(p1.p)
        n2 = len(p2.p)
        con = {}
        for i in range(n1):
            for j in range(n2):
                con[(i, j)] = np.ones((n1, n2), dtype=int)  # Simplified
        return con
    
    def _init_closed(self, g, p1, p2, *args, **kwargs):
        """Deal with closed argument of COMPARTICLE objects."""
        # Full particle in case of mirror symmetry
        if hasattr(p1, 'sym'):
            full1 = p1.pfull
        else:
            full1 = p1
        
        # For a closed particle the surface integral of -F should give 2*pi
        # See R. Fuchs and S. H. Liu, Phys. Rev. B 14, 5521 (1976)
        if (hasattr(full1, 'closed') and full1 == p2 and 
            hasattr(full1, 'closed') and any(full1.closed)):
            
            # Loop over particles
            for i in range(len(p1.p)):
                # Index to particle faces
                ind = p1.index(i)
                
                # Select particle and closed particle surface
                part = p1.p[i]
                full, direction, loc = self._closed_particle(p1, i)
                
                if full is not None:
                    if loc is not None:
                        # Use already computed Green function object
                        f = self._fun(g, loc, ind, *args, **kwargs)
                    else:
                        # Set up Green function
                        if args or kwargs:
                            options = BemOptions(*args, waitbar=0, **kwargs)
                            gstat = GreenStat(full, part, options)
                        else:
                            gstat = GreenStat(full, part)
                        
                        # Sum over closed surface
                        f = self._fun(gstat, *args, **kwargs)
                    
                    # Set diagonal elements of Green function
                    if g.deriv == 'norm':
                        g.set_diagonal(ind, -2 * np.pi * direction - f.T)
                    else:
                        diagonal_val = part.nvec * (-2 * np.pi * direction - f.T)
                        g.set_diagonal(ind, diagonal_val)
        
        return g
    
    def _closed_particle(self, p1, i):
        """Get closed particle information."""
        # This would need to be implemented based on the specific structure
        return None, None, None
    
    def _fun(self, g, *args, **kwargs):
        """Sum over closed surface."""
        p1 = g.p1
        p2 = g.p2
        
        # Deal with different calling sequences
        if args and isinstance(args[0], (int, np.integer, list, np.ndarray)):
            ind1, ind2 = args[0], args[1]
            args = args[2:]
        else:
            ind1 = np.arange(p1.n)
            ind2 = np.arange(p2.n)
        
        # Allocate output
        f = []
        
        # Initialize slicer object
        s = Slicer([p1.n, p2.n], ind1, ind2, *args, **kwargs)
        
        # Loop over slices
        for i in range(s.n):
            # Indices for slice
            ind, row, col = s(i)
            
            # Evaluate surface derivative of Green function
            if isinstance(g, GreenStat):
                F = g.eval(ind, 'F').reshape(len(row), len(col))
            elif isinstance(g, GreenRet):
                F = g.eval(ind, 0, 'F').reshape(len(row), len(col))
            
            # Sum over Green function elements
            area_factor = np.outer(p1.area[row], 1.0 / p2.area[col])
            f_slice = np.sum(area_factor * F, axis=0)
            f.append(f_slice)
        
        return np.concatenate(f) if f else np.array([])
    
    def eval(self, *args, **kwargs):
        """
        Evaluate retarded Green function.
        
        Parameters
        ----------
        *args : various
            i, j, key, enei for full matrix or i, j, key, enei, ind for selected elements
            
        Returns
        -------
        various
            Requested Green functions
        """
        if len(args) == 4:
            # Compute full matrix
            return self._eval1(*args, **kwargs)
        elif len(args) == 5:
            # Compute selected matrix elements
            return self._eval2(*args, **kwargs)
        else:
            raise ValueError("Invalid number of arguments")
    
    def _eval1(self, i, j, key, enei):
        """Evaluate retarded Green function (full matrix)."""
        # Evaluate connectivity matrix
        con = self.con[(i, j)]
        
        # Evaluate dielectric functions to get wavenumbers
        eps_vals = [eps(enei) for eps in self.p1.eps]
        k_vals = [np.sqrt(eps_val) * 2 * np.pi / enei for eps_val in eps_vals]
        
        # Evaluate G, F, H1, H2
        if key not in ['Gp', 'H1p', 'H2p']:
            # Allocate array
            g = np.zeros((self.p1.n, self.p2.n))
            
            # Loop over composite particles
            for i1 in range(con.shape[0]):
                for i2 in range(con.shape[1]):
                    if con[i1, i2]:
                        # Add Green function
                        g[np.ix_(self.p1.index(i1), self.p2.index(i2))] = \
                            self.g[(i1, i2)].eval(k_vals[con[i1, i2]], key)
        
        # Evaluate Gp, H1p, H2p
        else:
            # Allocate array
            g = np.zeros((self.p1.n, 3, self.p2.n))
            
            # Loop over composite particles
            for i1 in range(con.shape[0]):
                for i2 in range(con.shape[1]):
                    if con[i1, i2]:
                        # Add Green function
                        g[np.ix_(self.p1.index(i1), slice(None), self.p2.index(i2))] = \
                            self.g[(i1, i2)].eval(k_vals[con[i1, i2]], key)
        
        if np.all(g == 0):
            g = 0
        
        return g
    
    def _eval2(self, i, j, key, enei, ind):
        """Evaluate retarded Green function (selected matrix elements)."""
        # Evaluate connectivity matrix
        con = self.con[(i, j)]
        
        # Convert total index to cell array of subindices
        sub, ind_sub = self.block.ind2sub(ind)
        
        # Evaluate dielectric functions to get wavenumbers
        eps_vals = [eps(enei) for eps in self.p1.eps]
        k_vals = [np.sqrt(eps_val) * 2 * np.pi / enei for eps_val in eps_vals]
        
        # Place wavevectors into cell array
        con_k = con.copy().astype(float)
        con_k[con == 0] = np.nan
        con_k[~np.isnan(con_k)] = [k_vals[int(c)] for c in con_k[~np.isnan(con_k)]]
        
        # Evaluate Green function submatrices
        g_sub = {}
        for key_sub, (g_obj, k_val, sub_val) in zip(sub.keys(), 
                                                    zip(self.g.values(), con_k.flat, sub.values())):
            g_sub[key_sub] = self._fun_eval(g_obj, sub_val, k_val, key)
        
        # Assemble submatrices
        g = self.block.accumarray(ind_sub, g_sub)
        
        return g
    
    def _fun_eval(self, g, ind, k, key):
        """Evaluate Green function submatrices."""
        if np.isnan(k):
            if key in ['G', 'F', 'H1', 'H2']:
                return np.zeros((len(ind), 1))
            else:
                return np.zeros((len(ind), 3))
        else:
            return g.eval(ind, k, key)
    
    def field(self, sig, inout=1):
        """
        Electric and magnetic field inside/outside of particle surface.
        
        Parameters
        ----------
        sig : CompStruct
            Surface charges & currents (see bemret)
        inout : int, optional
            Fields inside (inout=1, default) or outside (inout=2) of particle surface
            
        Returns
        -------
        CompStruct
            Electric and magnetic fields
        """
        # Wavelength and wavenumber of light in vacuum
        enei = sig.enei
        k = 2 * np.pi / sig.enei
        
        # Green function and E = i k A
        e = (1j * k * (np.matmul(self.eval(inout, 1, 'G', enei), sig.h1) +
                      np.matmul(self.eval(inout, 2, 'G', enei), sig.h2)))
        
        # Derivative of Green function
        if inout == 1:
            H1p = self.eval(inout, 1, 'H1p', enei)
            H2p = self.eval(inout, 2, 'H1p', enei)
        else:
            H1p = self.eval(inout, 1, 'H2p', enei)
            H2p = self.eval(inout, 2, 'H2p', enei)
        
        # Add derivative of scalar potential to electric field
        e = e - np.matmul(H1p, sig.sig1) - np.matmul(H2p, sig.sig2)
        
        # Magnetic field
        h = self._cross(H1p, sig.h1) + self._cross(H2p, sig.h2)
        
        # Set output
        return CompStruct(self.p1, enei, e=e, h=h)
    
    def _cross(self, G, h):
        """Multidimensional cross product."""
        if np.isscalar(G):
            return 0
        
        # Size of vector field
        siz = h.shape
        siz = (siz[0], *siz[2:]) if len(siz) > 2 else (siz[0], 1)
        
        # Get component
        def at(h, i):
            return h[:, i, :].reshape(siz)
        
        # Cross product
        cross = np.concatenate([
            np.matmul(G[:, 1, :], at(h, 2)) - np.matmul(G[:, 2, :], at(h, 1)),
            np.matmul(G[:, 2, :], at(h, 0)) - np.matmul(G[:, 0, :], at(h, 2)),
            np.matmul(G[:, 0, :], at(h, 1)) - np.matmul(G[:, 1, :], at(h, 0))
        ], axis=1)
        
        return cross
    
    def potential(self, sig, inout=1, *args, **kwargs):
        """
        Potentials and surface derivatives inside/outside of particle.
        
        Parameters
        ----------
        sig : CompStruct
            Surface charges (see bemstat)
        inout : int, optional
            Potentials inside (inout=1, default) or outside (inout=2) of particle
        *args, **kwargs : optional
            Additional arguments to Green function
            
        Returns
        -------
        CompStruct
            Potentials and surface derivatives
        """
        enei = sig.enei
        var = [enei] + list(args)
        
        # Set parameters that depend on inside/outside
        H_key = 'H1' if inout == 1 else 'H2'
        
        # Green functions
        G1 = self.__getitem__(inout, 1).G(*var, **kwargs)
        G2 = self.__getitem__(inout, 2).G(*var, **kwargs)
        
        # Surface derivatives of Green functions
        H1 = getattr(self.__getitem__(inout, 1), H_key)(*var, **kwargs)
        H2 = getattr(self.__getitem__(inout, 2), H_key)(*var, **kwargs)
        
        # Potential and surface derivative
        # Scalar potential
        phi = np.matmul(G1, sig.sig1) + np.matmul(G2, sig.sig2)
        phip = np.matmul(H1, sig.sig1) + np.matmul(H2, sig.sig2)
        
        # Vector potential
        a = np.matmul(G1, sig.h1) + np.matmul(G2, sig.h2)
        ap = np.matmul(H1, sig.h1) + np.matmul(H2, sig.h2)
        
        # Set output
        if inout == 1:
            return CompStruct(self.p1, enei, phi1=phi, phi1p=phip, a1=a, a1p=ap)
        else:
            return CompStruct(self.p1, enei, phi2=phi, phi2p=phip, a2=a, a2p=ap)
    
    def __getitem__(self, *indices):
        """
        Derived properties for CompGreenRet objects.
        
        Usage:
            obj[i, j].G(enei) : composite Green function
        """
        return self.eval(*indices)
    
    def __getattr__(self, name):
        """Handle attribute access."""
        if name == 'deriv':
            return next(iter(self.g.values())).deriv if self.g else None
        else:
            return super().__getattribute__(name)