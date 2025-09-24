"""
Green function for particle using full Maxwell's equations and ACA.

This is a Python translation of the MATLAB @compgreenret class from MNPBEM.
"""
import numpy as np
from typing import Optional, Union, Any, Dict, List, Tuple

# Import from existing PyMNPBEM modules
from ..hmatrices.clustertree import ClusterTree
from ..hmatrices.hmatrix import HMatrix
from ..compgreenret import CompGreenRet as BaseCompGreenRet
from ...Particles.compstruct import CompStruct
from ...Particles.comparticle import CompArticle
from ...mex.hmatgreenret import hmatgreenret


class CompGreenRet:
    """
    Green function for particle using full Maxwell's equations and ACA.
    
    Properties:
        p (CompArticle): COMPARTICLE object
        g (BaseCompGreenRet): Green functions connecting particle boundaries  
        hmat (HMatrix): template for H-matrix
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize Green functions for composite objects.
        
        Usage:
            obj = CompGreenRet(p, op)
            
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
        return f"aca.CompGreenRet(p={self.p}, g={self.g}, hmat={self.hmat})"
    
    def _init(self, p: CompArticle, *args, **kwargs):
        """
        Initialize composite Green function.
        
        Args:
            p: COMPARTICLE object
            *args, **kwargs: additional arguments
        """
        # Save particle
        self.p = p
        
        # Initialize COMPGREEN object (using base implementation)
        self.g = BaseCompGreenRet(p, p, *args, **kwargs)
        
        # Make cluster tree
        tree = ClusterTree(p, *args, **kwargs)
        
        # Template for H-matrix
        self.hmat = HMatrix(tree, *args, **kwargs)
    
    def eval(self, i: int, j: int, key: str, enei: float) -> HMatrix:
        """
        Evaluate retarded Green function.
        
        Deals with calls to obj = aca.compgreenret:
            g = obj[i, j].G(enei)
            
        Args:
            i, j: indices
            key: str - 'G' (Green function), 'F' (Surface derivative of Green function),
                      'H1' (F + 2*pi), 'H2' (F - 2*pi)
            enei: light wavelength in vacuum
            
        Returns:
            hmat: evaluated H-matrix
        """
        p, hmat = self.p, self.hmat
        
        # Fill full matrices
        def fun(row, col):
            # Convert to linear index (MATLAB's sub2ind equivalent)
            linear_idx = np.ravel_multi_index((row-1, col-1), (p.n, p.n))
            return self.g.eval(i, j, key, enei, linear_idx)
        
        # Compute full matrices
        hmat = hmat.fillval(fun)
        
        # Size of clusters
        tree = hmat.tree
        siz = tree.cind[:, 1] - tree.cind[:, 0] + 1
        
        # Allocate low-rank matrices
        hmat.lhs = [np.zeros((s, 1)) for s in siz[hmat.row2]]
        hmat.rhs = [np.zeros((s, 1)) for s in siz[hmat.col2]]
        
        # Connectivity matrix
        con = self.g.con[i][j].copy()
        
        # Evaluate dielectric functions to get wavenumbers
        k_values = []
        for eps_func in self.p.eps:
            _, k = eps_func(enei)
            k_values.append(k)
        k = np.array(k_values)
        
        # Place wavevectors into connectivity matrix
        con[con == 0] = np.nan
        con_mask = ~np.isnan(con)
        con[con_mask] = k[con[con_mask].astype(int)]
        
        # Particle structure for computation
        ind = hmat.tree.ind[:, 0]
        pmex = {
            'pos': p.pos[ind, :],
            'nvec': p.nvec[ind, :],
            'area': p.area[ind]
        }
        
        # Tree indices and options
        tmex = self._treemex(hmat)
        op = {
            'htol': hmat.htol,
            'kmax': hmat.kmax
        }
        
        # Main computation loop
        for i_idx in range(con.shape[0]):
            for j_idx in range(con.shape[1]):
                if not np.isnan(con[i_idx, j_idx]):
                    ii = self._uintmex(i_idx + 1)  # Convert to 1-based indexing
                    jj = self._uintmex(j_idx + 1)
                    
                    # Compute low-rank matrix using ACA
                    if key == 'G':
                        L, R = hmatgreenret(pmex, tmex, 'G', ii, jj, 
                                          complex(con[i_idx, j_idx]), op)
                    elif key in ['F', 'H1', 'H2']:
                        L, R = hmatgreenret(pmex, tmex, 'F', ii, jj, 
                                          complex(con[i_idx, j_idx]), op)
                    
                    # Index to low-rank matrices
                    ind = [x is not None and x.size > 0 for x in L]
                    
                    # Set low-rank matrices
                    for idx, is_valid in enumerate(ind):
                        if is_valid:
                            hmat.lhs[idx] = L[idx]
                            hmat.rhs[idx] = R[idx]
        
        return hmat
    
    def potential(self, sig: CompStruct, inout: int = 1, *args, **kwargs) -> CompStruct:
        """
        Potentials and surface derivatives inside/outside of particle.
        Computed from solutions of full Maxwell equations.
        
        Args:
            sig: CompStruct with surface charges (see bemstat)
            inout: potentials inside (inout=1, default) or outside (inout=2) of particle
            *args, **kwargs: additional arguments passed to Green function
            
        Returns:
            pot: CompStruct object with potentials & surface derivatives
        """
        enei = sig.enei
        var = [enei] + list(args) + list(kwargs.values())
        
        # Set parameters that depend on inside/outside
        H = 'H1' if inout == 1 else 'H2'
        
        # Green functions
        G1 = self[inout, 1].G(enei, *args, **kwargs)
        G2 = self[inout, 2].G(enei, *args, **kwargs)
        
        # Surface derivatives of Green functions
        H1 = getattr(self[inout, 1], H)(enei, *args, **kwargs)
        H2 = getattr(self[inout, 2], H)(enei, *args, **kwargs)
        
        # Matrix multiplication helper
        def matmul(x, y):
            y_reshaped = y.reshape(y.shape[0], -1)
            result = x @ y_reshaped
            return result.reshape(y.shape)
        
        # Scalar potential
        phi = matmul(G1, sig.sig1) + matmul(G2, sig.sig2)
        phip = matmul(H1, sig.sig1) + matmul(H2, sig.sig2)
        
        # Vector potential
        a = matmul(G1, sig.h1) + matmul(G2, sig.h2)
        ap = matmul(H1, sig.h1) + matmul(H2, sig.h2)
        
        # Set output
        if inout == 1:
            pot = CompStruct(self.p, enei, 
                           phi1=phi, phi1p=phip, a1=a, a1p=ap)
        else:
            pot = CompStruct(self.p, enei,
                           phi2=phi, phi2p=phip, a2=a, a2p=ap)
        
        return pot
    
    def __getitem__(self, key):
        """
        Handle subscript access obj[i, j] for Green function calls.
        Implements MATLAB's obj{i,j} syntax.
        """
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            return GreenFunctionAccessor(self, i, j)
        else:
            raise IndexError("Expected tuple of length 2 for Green function access")
    
    def _treemex(self, hmat: HMatrix) -> Dict:
        """
        Convert tree to format expected by MEX function.
        
        Args:
            hmat: H-matrix object
            
        Returns:
            dict: Tree data in MEX-compatible format
        """
        tree = hmat.tree
        return {
            'ind': tree.ind,
            'cind': tree.cind,
            'row2': hmat.row2,
            'col2': hmat.col2
        }
    
    def _uintmex(self, val: int) -> int:
        """
        Convert to unsigned integer for MEX compatibility.
        
        Args:
            val: integer value
            
        Returns:
            int: converted value
        """
        return int(val)


class GreenFunctionAccessor:
    """
    Helper class to handle obj[i,j].method() syntax from MATLAB.
    Implements MATLAB's subsref functionality for Green functions.
    """
    
    def __init__(self, parent: CompGreenRet, i: int, j: int):
        self.parent = parent
        self.i = i
        self.j = j
    
    def G(self, enei: float, *args, **kwargs) -> np.ndarray:
        """Green function."""
        return self.parent.eval(self.i, self.j, 'G', enei)
    
    def F(self, enei: float, *args, **kwargs) -> np.ndarray:
        """Surface derivative of Green function."""
        return self.parent.eval(self.i, self.j, 'F', enei)
    
    def H1(self, enei: float, *args, **kwargs) -> np.ndarray:
        """F + 2*pi."""
        return self.parent.eval(self.i, self.j, 'H1', enei)
    
    def H2(self, enei: float, *args, **kwargs) -> np.ndarray:
        """F - 2*pi."""
        return self.parent.eval(self.i, self.j, 'H2', enei)


# Module-level function for compatibility with MATLAB-style function calls
def compgreenret(*args, **kwargs) -> CompGreenRet:
    """
    Create CompGreenRet object - for compatibility with existing code.
    
    This function provides MATLAB-style constructor syntax:
        g = compgreenret(p, options)
    instead of:
        g = CompGreenRet(p, options)
    """
    return CompGreenRet(*args, **kwargs)