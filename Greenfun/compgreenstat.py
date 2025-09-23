import numpy as np
from abc import ABC, abstractmethod
from .bembase import BemBase
from .greenstat import GreenStat
from .compstruct import CompStruct
from .slicer import Slicer
from .bemoptions import BemOptions


class CompGreenStat(BemBase):
    """Green function for composite points and particle in quasistatic approximation."""
    
    # Class attributes
    name = 'greenfunction'
    needs = [{'sim': 'stat'}]
    
    def __init__(self, p1, p2, *args, **kwargs):
        """
        Initialize Green functions for composite objects.
        
        Parameters
        ----------
        p1 : object
            Green function between points p1 and comparticle p2
        p2 : object
            Green function between points p1 and comparticle p2
        *args, **kwargs : optional
            Additional options (see BemOptions)
        """
        super().__init__()
        self.p1 = None
        self.p2 = None
        self.g = None
        self._init(p1, p2, *args, **kwargs)
    
    def __str__(self):
        """Command window display."""
        info = {
            'p1': self.p1,
            'p2': self.p2,
            'g': self.g
        }
        return f"CompGreenStat:\n{info}"
    
    def _init(self, p1, p2, *args, **kwargs):
        """Initialize composite Green function."""
        self.p1 = p1
        pp1 = p1.p
        self.p2 = p2
        pp2 = p2.p
        
        # Initialize Green function
        p1_combined = np.vstack([p for p in pp1])
        p2_combined = np.vstack([p for p in pp2])
        self.g = GreenStat(p1_combined, p2_combined, *args, **kwargs)
        
        # Full particle in case of mirror symmetry
        if hasattr(p1, 'sym'):
            full1 = p1.pfull
        else:
            full1 = p1
        
        # For a closed particle the surface integral of -F should give 2*pi,
        # see R. Fuchs and S. H. Liu, Phys. Rev. B 14, 5521 (1976).
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
                        f = self._fun(self.g, loc, ind, *args, **kwargs)
                    else:
                        # Set up Green function
                        if args or kwargs:
                            options = BemOptions(*args, waitbar=0, **kwargs)
                            g = GreenStat(full, part, options)
                        else:
                            g = GreenStat(full, part)
                        
                        # Sum over closed surface
                        f = self._fun(g, *args, **kwargs)
                    
                    # Set diagonal elements of Green function
                    if self.g.deriv == 'norm':
                        self.g.set_diagonal(ind, -2 * np.pi * direction - f.T)
                    else:
                        diagonal_val = part.nvec * (-2 * np.pi * direction - f.T)
                        self.g.set_diagonal(ind, diagonal_val)
    
    def _closed_particle(self, p1, i):
        """Get closed particle information."""
        # This would need to be implemented based on the specific structure
        # of your particle objects
        # Returns full particle, direction, and location
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
            F = g.eval(ind, 'F').reshape(len(row), len(col))
            
            # Sum over Green function elements
            area_factor = np.outer(p1.area[row], 1.0 / p2.area[col])
            f_slice = np.sum(area_factor * F, axis=0)
            f.append(f_slice)
        
        return np.concatenate(f) if f else np.array([])
    
    def eval(self, *args, **kwargs):
        """
        Evaluate Green function.
        
        Parameters
        ----------
        *args : various
            Can include index and keys like 'G', 'F', 'H1', 'H2', 'Gp', 'H1p', 'H2p'
            
        Returns
        -------
        tuple
            Requested Green functions
        """
        return self.g.eval(*args, **kwargs)
    
    def field(self, sig, inout=1):
        """
        Electric field inside/outside of particle surface.
        
        Parameters
        ----------
        sig : CompStruct
            Surface charges (see BemStat)
        inout : int, optional
            Fields inside (inout=1, default) or outside (inout=2) of particle surface
            
        Returns
        -------
        CompStruct
            Electric field
        """
        # Derivative of Green function
        if inout == 1:
            Hp = self.g.eval('H1p')
        else:
            Hp = self.g.eval('H2p')
        
        # Electric field
        e = -np.matmul(Hp, sig.sig)
        
        # Set output
        return CompStruct(self.p1, sig.enei, e=e)
    
    def potential(self, sig, inout=1):
        """
        Determine potentials and surface derivatives inside/outside of particle.
        
        Parameters
        ----------
        sig : CompStruct
            Surface charges (see BemStat)
        inout : int, optional
            Potentials inside (inout=1, default) or outside (inout=2) of particle surface
            
        Returns
        -------
        CompStruct
            Potentials and surface derivatives
        """
        # Set parameters that depend on inside/outside
        H_key = 'H1' if inout == 1 else 'H2'
        
        # Get Green function and surface derivative
        G, H = self.g.eval('G', H_key)
        
        # Potential and surface derivative
        phi = np.matmul(G, sig.sig)
        phip = np.matmul(H, sig.sig)
        
        # Set output
        if inout == 1:
            return CompStruct(self.p1, sig.enei, phi1=phi, phi1p=phip)
        else:
            return CompStruct(self.p1, sig.enei, phi2=phi, phi2p=phip)
    
    def __getattr__(self, name):
        """
        Derived properties for CompGreenStat objects.
        
        Supports accessing Green function components and methods.
        """
        if name in ['G', 'F', 'H1', 'H2', 'Gp']:
            return self.g.eval(name)
        elif name == 'deriv':
            return self.g.deriv
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")