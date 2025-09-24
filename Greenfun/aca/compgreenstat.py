"""
Green function for particle in quasistatic approximation using ACA.

This is a Python translation of the MATLAB @compgreenstat class from MNPBEM.
"""
import numpy as np
from typing import Optional, Union, Any, Dict, List, Tuple

# Import from existing PyMNPBEM modules
from ..hmatrices.clustertree import ClusterTree
from ..hmatrices.hmatrix import HMatrix
from ..compgreenstat import CompGreenStat as BaseCompGreenStat
from ...Particles.compstruct import CompStruct
from ...Particles.comparticle import CompArticle
from ...mex.hmatgreenstat import hmatgreenstat
from ...utils.treemex import treemex


class CompGreenStat:
    """
    Green function for particle in quasistatic approximation using ACA.
    
    Properties:
        p (CompArticle): COMPARTICLE object
        g (BaseCompGreenStat): COMPGREENSTAT object
        hmat (HMatrix): template for H-matrix
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize Green function for COMPARTICLE.
        
        Usage:
            obj = CompGreenStat(p, op)
            
        Input:
            p: COMPARTICLE object
            op: options (see BEMOPTIONS)
        """
        self.p = None
        self.g = None
        self.hmat = None
        
        # Initialize the object
        self._init(*args, **kwargs)
    
    def __repr__(self):
        """String representation of the object."""
        return f"aca.CompGreenStat(p={self.p}, g={self.g}, hmat={self.hmat})"
    
    def _init(self, p: CompArticle, *args, **kwargs):
        """
        Initialize composite Green function.
        
        Args:
            p: COMPARTICLE object
            *args, **kwargs: additional arguments
        """
        # Save particle
        self.p = p
        
        # Initialize COMPGREEN object
        self.g = BaseCompGreenStat(p, p, *args, **kwargs)
        
        # Make cluster tree
        tree = ClusterTree(p, *args, **kwargs)
        
        # Template for H-matrix
        self.hmat = HMatrix(tree, *args, **kwargs)
    
    def eval(self, *args) -> Union[HMatrix, Tuple[HMatrix, ...]]:
        """
        Evaluate Green function.
        
        Usage for obj = aca.compgreenstat:
            varargout = eval(obj, key1, key2, ...)
            
        Input:
            key: 'G' - Green function
                 'F' - Surface derivative of Green function
                 'H1' - F + 2*pi
                 'H2' - F - 2*pi
                 
        Output:
            varargout: requested Green functions
        """
        # Particle
        p = self.p
        
        # Options for ACA
        hmat = self.hmat
        op = {
            'htol': min(hmat.htol) if hasattr(hmat.htol, '__iter__') else hmat.htol,
            'kmax': max(hmat.kmax) if hasattr(hmat.kmax, '__iter__') else hmat.kmax
        }
        
        # Assign output
        varargout = []
        
        for i, key in enumerate(args):
            # Check for input
            if key == 'Gp':
                raise ValueError('Gp not implemented for aca.compgreenstat')
            
            # Fill full matrices
            def fun(row, col):
                linear_idx = np.ravel_multi_index((row-1, col-1), (p.n, p.n))
                return self.g.eval(linear_idx, key)
            
            # Compute full matrices
            hmat_copy = hmat.fillval(fun)
            
            # Particle structure for MEX function call
            ind = hmat.tree.ind[:, 0]
            pmex = {
                'pos': p.pos[ind, :],
                'nvec': p.nvec[ind, :],
                'area': p.area[ind]
            }
            
            # Tree and cluster indices for MEX function call
            tmex = treemex(hmat)
            
            # Compute low-rank approximation
            if key == 'G':
                hmat_copy.lhs, hmat_copy.rhs = hmatgreenstat(pmex, tmex, 'G', op)
            elif key in ['F', 'H1', 'H2']:
                hmat_copy.lhs, hmat_copy.rhs = hmatgreenstat(pmex, tmex, 'F', op)
            
            # Assign output
            varargout.append(hmat_copy)
        
        # Return single value if only one argument, tuple otherwise
        if len(varargout) == 1:
            return varargout[0]
        else:
            return tuple(varargout)
    
    def potential(self, sig: CompStruct, inout: int = 1) -> CompStruct:
        """
        Determine potentials and surface derivatives inside/outside of particle.
        
        Usage for obj = aca.compgreenstat:
            pot = potential(obj, sig, inout)
            
        Input:
            sig: compstruct with surface charges (see BEMSTAT)
            inout: potentials inside (inout=1, default) or 
                   outside (inout=2) of particle surface
                   
        Output:
            pot: compstruct object with potentials & surface derivatives
        """
        # Set parameters that depend on inside/outside
        H = 'H1' if inout == 1 else 'H2'
        
        # Get Green function and surface derivative
        G, H_mat = self.eval('G', H)
        
        # Matrix multiplication helper
        def matmul(x, y):
            y_reshaped = y.reshape(y.shape[0], -1)
            result = x @ y_reshaped
            return result.reshape(y.shape)
        
        # Potential and surface derivative
        phi = matmul(G, sig.sig)
        phip = matmul(H_mat, sig.sig)
        
        # Set output
        if inout == 1:
            pot = CompStruct(self.p, sig.enei, phi1=phi, phi1p=phip)
        else:
            pot = CompStruct(self.p, sig.enei, phi2=phi, phi2p=phip)
        
        return pot
    
    def __getattr__(self, name: str):
        """
        Handle attribute access for Green function properties.
        
        Usage for obj = aca.compgreenstat:
            obj.G                  : Green function
            obj.F, obj.H1, obj.H2  : Surface derivatives
        """
        if name in ['G', 'F', 'H1', 'H2']:
            return self.eval(name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


# Module-level function for compatibility with MATLAB-style function calls
def compgreenstat(*args, **kwargs) -> CompGreenStat:
    """
    Create CompGreenStat object - for compatibility with existing code.
    
    This function provides MATLAB-style constructor syntax:
        g = compgreenstat(p, options)
    instead of:
        g = CompGreenStat(p, options)
    """
    return CompGreenStat(*args, **kwargs)