"""
Green function for particle and layer structure using full Maxwell's equations and ACA.

This is a Python translation of the MATLAB @compgreenretlayer class from MNPBEM.
"""
import numpy as np
from typing import Optional, Union, Any, Dict, List, Tuple

# Import from existing PyMNPBEM modules
from ..hmatrices.clustertree import ClusterTree
from ..hmatrices.hmatrix import HMatrix
from ..compgreenretlayer import CompGreenRetLayer as BaseCompGreenRetLayer
from ...Particles.compstruct import CompStruct
from ...Particles.comparticle import CompArticle
from ...mex.hmatgreenret import hmatgreenret
from ...mex.hmatgreentab1 import hmatgreentab1
from ...mex.hmatgreentab2 import hmatgreentab2
from ...utils.bemoptions import getbemoptions
from ...utils.matindex import matindex
from ...utils.treemex import treemex
from ...utils.uintmex import uintmex
from ...layer.indlayer import indlayer
from ...layer.mindist import mindist


class CompGreenRetLayer:
    """
    Green function for particle and layer structure using full Maxwell's equations and ACA.
    
    Properties:
        p (CompArticle): COMPARTICLE object
        layer (object): layer structure
        g (BaseCompGreenRetLayer): Green function connecting particle boundaries
        hmat (HMatrix): template for H-matrix
        ind (np.ndarray): starting cluster index for given particle
        rmod (str): 'log' for logspace r-table or 'lin' for linspace
        zmod (str): 'log' for logspace z-table or 'lin' for linspace
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize Green functions for composite objects.
        
        Usage:
            obj = CompGreenRetLayer(p, op)
            
        Input:
            p: COMPARTICLE object
            op: options (see BEMOPTIONS)
        """
        self.p = None
        self.layer = None
        self.g = None
        self.hmat = None
        self.ind = None
        self.rmod = 'log'
        self.zmod = 'log'
        
        # Initialize the object
        self._init(*args, **kwargs)
    
    def __repr__(self):
        """String representation of the object."""
        return f"aca.CompGreenRetLayer(p={self.p}, layer={self.layer}, g={self.g}, hmat={self.hmat})"
    
    def _init(self, p: CompArticle, *args, **kwargs):
        """
        Initialize composite Green function.
        
        Args:
            p: COMPARTICLE object
            *args, **kwargs: additional arguments
        """
        op = getbemoptions(*args, **kwargs)
        
        # Save particle and layer structure
        self.p = p
        self.layer = op.layer
        
        # Grid for tabulated Green functions
        if hasattr(op, 'rmod'):
            self.rmod = op.rmod
        if hasattr(op, 'zmod'):
            self.zmod = op.zmod
            
        # Initialize COMPGREEN object
        self.g = BaseCompGreenRetLayer(p, p, *args, **kwargs)
        
        # Make cluster tree
        tree = ClusterTree(p, *args, **kwargs)
        
        # Template for H-matrix
        self.hmat = HMatrix(tree, *args, **kwargs)
        
        # Particle index for cluster
        # TODO: Implement when ipart function is found
        ind1 = self._ipart_placeholder(p, tree.ind[tree.cind[:, 0], 0].reshape(1, -1))
        ind2 = self._ipart_placeholder(p, tree.ind[tree.cind[:, 1], 0].reshape(1, -1))
        
        # Cluster index
        self.ind = np.zeros(p.np, dtype=int)
        for i in range(p.np):
            mask = (ind1 == i) & (ind2 == i)
            idx = np.where(mask)[0]
            if len(idx) > 0:
                self.ind[i] = idx[0]
    
    def eval(self, i: int, j: int, key: str, enei: float) -> Any:
        """
        Evaluate retarded Green function for layer structure.
        
        Args:
            i, j: indices
            key: str - 'G', 'F', 'H1', 'H2'
            enei: light wavelength in vacuum
            
        Returns:
            hmat: evaluated H-matrix
        """
        # Depending on i and j, the Green function interaction can be only direct
        # or additionally influenced by layer reflections
        if not (i == 2 and j == 2):
            return self._eval1(i, j, key, enei)
        else:
            return self._eval2(i, j, key, enei)
    
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
        """Handle subscript access obj[i, j] for Green function calls."""
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            return GreenFunctionAccessor(self, i, j)
        else:
            raise IndexError("Expected tuple of length 2 for Green function access")
    
    def _eval1(self, i: int, j: int, key: str, enei: float) -> HMatrix:
        """
        Evaluate retarded Green function for layer structure (direct).
        """
        p, hmat = self.p, self.hmat
        
        # Fill full matrices
        def fun(row, col):
            linear_idx = np.ravel_multi_index((row-1, col-1), (p.n, p.n))
            return self.g.eval(i, j, key, enei, linear_idx)
        
        # Compute full matrices
        hmat = hmat.fillval(fun)
        
        # Compute low-rank matrices
        hmat.lhs, hmat.rhs = self._lowrank1(i, j, key, enei)
        
        return hmat
    
    def _eval2(self, i: int, j: int, key: str, enei: float) -> Dict:
        """
        Evaluate retarded Green function for layer structure (reflected).
        """
        hmat = self.hmat
        
        # Cluster tree
        tree = hmat.tree
        
        # Matrix indices for full matrices
        ind = []
        siz = []
        n = []
        
        for row, col in zip(hmat.row1, hmat.col1):
            idx, s, nn = matindex(tree, tree, row, col)
            ind.append(idx)
            siz.append(s)
            n.append(nn)
        
        # Full matrices of Green function for layer structure
        val = self.g.eval(i, j, key, enei, np.vstack(ind))
        
        # Low-rank matrices for direct Green function interaction
        lhs, rhs = self._lowrank1(i, j, key, enei)
        
        g = {}
        
        # Loop over field names
        for name in val.keys():
            # Fill full matrices
            mat = np.split(val[name], np.cumsum([nn for nn in n])[:-1])
            val1 = [mat_i.reshape(siz_i) for mat_i, siz_i in zip(mat, siz)]
            
            # Fill low-rank matrices (direct)
            if name in ['p', 'ss', 'hh']:
                lhs1, rhs1 = lhs, rhs
            else:
                lhs1 = [np.zeros((lhs_i.shape[0], 1)) for lhs_i in lhs]
                rhs1 = [np.zeros((rhs_i.shape[0], 1)) for rhs_i in rhs]
            
            # Fill low-rank matrix (reflected)
            lhs2, rhs2 = self._lowrank2(key, name, enei)
            
            # Assign output
            g[name] = hmat.copy()
            
            # Set full matrices
            g[name].val = val1
            
            # Set low-rank matrices
            g[name].lhs = [np.hstack([lhs1_i, lhs2_i]) for lhs1_i, lhs2_i in zip(lhs1, lhs2)]
            g[name].rhs = [np.hstack([rhs1_i, rhs2_i]) for rhs1_i, rhs2_i in zip(rhs1, rhs2)]
            
            # Truncate matrix
            g[name] = g[name].truncate(hmat.htol)
        
        return g
    
    def _lowrank1(self, i: int, j: int, key: str, enei: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Evaluate low-rank matrices for layer structure (direct).
        """
        p, hmat = self.p, self.hmat
        
        # Size of clusters
        tree = hmat.tree
        siz = tree.cind[:, 1] - tree.cind[:, 0] + 1
        
        # Allocate low-rank matrices
        lhs = [np.zeros((s, 1)) for s in siz[hmat.row2]]
        rhs = [np.zeros((s, 1)) for s in siz[hmat.col2]]
        
        # Connectivity matrix
        con = self.g.g.con[i][j].copy()
        
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
        
        # Particle structure for MEX function call
        ind = hmat.tree.ind[:, 0]
        pmex = {
            'pos': p.pos[ind, :],
            'nvec': p.nvec[ind, :],
            'area': p.area[ind]
        }
        
        # Tree indices and options for MEX function call
        tmex = treemex(hmat)
        op = {'htol': hmat.htol, 'kmax': hmat.kmax}
        
        for i_idx in range(con.shape[0]):
            for j_idx in range(con.shape[1]):
                if not np.isnan(con[i_idx, j_idx]):
                    # Starting cluster
                    row = uintmex(self.ind[i_idx])
                    col = uintmex(self.ind[j_idx])
                    
                    # Compute low-rank matrix using ACA
                    if key == 'G':
                        L, R = hmatgreenret(pmex, tmex, 'G', row, col, 
                                          complex(con[i_idx, j_idx]), op)
                    elif key in ['F', 'H1', 'H2']:
                        L, R = hmatgreenret(pmex, tmex, 'F', row, col,
                                          complex(con[i_idx, j_idx]), op)
                    
                    # Index to low-rank matrices
                    ind_valid = [x is not None and x.size > 0 for x in L]
                    
                    # Set low-rank matrices
                    for idx, is_valid in enumerate(ind_valid):
                        if is_valid:
                            lhs[idx] = L[idx]
                            rhs[idx] = R[idx]
        
        return lhs, rhs
    
    def _lowrank2(self, key: str, name: str, enei: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Evaluate low-rank matrices for layer structure (reflected).
        """
        p, hmat = self.p, self.hmat
        
        # Size of clusters
        tree = hmat.tree
        siz = tree.cind[:, 1] - tree.cind[:, 0] + 1
        
        # Allocate low-rank matrices
        lhs = [np.zeros((s, 1)) for s in siz[hmat.row2]]
        rhs = [np.zeros((s, 1)) for s in siz[hmat.col2]]
        
        # COMPGREENTABLAYER object
        # Table of reflected Green function, use INSIDE to select cell index
        gtab = self.g.gr.tab.eval(enei)
        
        # Multiply Green function with distance-dependent factors
        # TODO: Implement when norm function is found
        g_vals, fr_vals, fz_vals = self._norm_placeholder(gtab.g)
        
        # Minimum distance to layer structure
        # TODO: Implement when round function is found
        z = mindist(self.layer, self._round_placeholder(self.layer, p.pos[:, 2]))
        
        # Particle structure for MEX function call
        ind = hmat.tree.ind[:, 0]
        pmex = {
            'pos': p.pos[ind, :],
            'nvec': p.nvec[ind, :],
            'area': p.area[ind],
            'z': z[ind]
        }
        
        # Tree indices and options for MEX function call
        tmex = treemex(hmat)
        op = {'htol': hmat.htol, 'kmax': hmat.kmax}
        
        for i1 in range(p.np):
            for i2 in range(p.np):
                # z-values of first boundary elements
                z1 = p.p[i1].pos[0, 2]
                z2 = p.p[i2].pos[0, 2]
                
                # Index to layers within which particles are embedded
                ind1 = uintmex(indlayer(self.layer, z1))
                ind2 = uintmex(indlayer(self.layer, z2))
                
                # Starting cluster
                row = uintmex(self.ind[i1])
                col = uintmex(self.ind[i2])
                
                # Reshape function
                fun = lambda x: x.flatten()
                
                # Compute low-rank matrix using ACA
                if key == 'G':
                    # Tabulated Green functions
                    # TODO: Implement when inside function is found
                    tab_g = gtab.g[self._inside_placeholder(gtab, 0, z1, z2)]
                    tab = {
                        'r': tab_g.r,
                        'rmod': self.rmod,
                        'z1': tab_g.z1,
                        'z2': tab_g.z2,
                        'zmod': self.zmod,
                        'G': fun(g_vals[name])
                    }
                    
                    # Compute Green function
                    L, R = hmatgreentab1(pmex, tmex, row, col, tab, ind1, ind2, op)
                
                elif key in ['F', 'H1', 'H2']:
                    # Tabulated surface derivatives of Green functions
                    # TODO: Implement when inside function is found
                    tab_g = gtab.g[self._inside_placeholder(gtab, 0, z1, z2)]
                    tab = {
                        'r': tab_g.r,
                        'rmod': self.rmod,
                        'z1': tab_g.z1,
                        'z2': tab_g.z2,
                        'zmod': self.zmod,
                        'Fr': fun(fr_vals[name]),
                        'Fz': fun(fz_vals[name])
                    }
                    
                    # Compute surface derivative of Green function
                    L, R = hmatgreentab2(pmex, tmex, row, col, tab, ind1, ind2, op)
                
                # Index to low-rank matrices
                ind_valid = [x is not None and x.size > 0 for x in L]
                
                # Set low-rank matrices
                for idx, is_valid in enumerate(ind_valid):
                    if is_valid:
                        lhs[idx] = L[idx]
                        rhs[idx] = R[idx]
        
        return lhs, rhs
    
    # Placeholder methods - TO BE IMPLEMENTED when MATLAB functions are found
    def _ipart_placeholder(self, p, indices):
        """
        TODO: Implement ipart function from MATLAB.
        Get particle indices for cluster tree.
        """
        # Temporary implementation - return zeros for now
        return np.zeros_like(indices)
    
    def _norm_placeholder(self, gtab_g_list):
        """
        TODO: Implement norm function from MATLAB.
        Normalize Green function with distance-dependent factors.
        """
        # Temporary implementation - return empty dicts
        g_vals = {}
        fr_vals = {}
        fz_vals = {}
        return g_vals, fr_vals, fz_vals
    
    def _round_placeholder(self, layer, z_vals):
        """
        TODO: Implement round function from MATLAB (layer-related).
        Round z-values to layer structure.
        """
        # Temporary implementation - just return z_vals
        return z_vals
    
    def _inside_placeholder(self, gtab, val, z1, z2):
        """
        TODO: Implement inside function from MATLAB.
        Get index for tabulated Green function based on z-coordinates.
        """
        # Temporary implementation - return 0
        return 0


class GreenFunctionAccessor:
    """
    Helper class to handle obj[i,j].method() syntax from MATLAB.
    """
    
    def __init__(self, parent: CompGreenRetLayer, i: int, j: int):
        self.parent = parent
        self.i = i
        self.j = j
    
    def G(self, enei: float, *args, **kwargs) -> Any:
        """Green function."""
        return self.parent.eval(self.i, self.j, 'G', enei)
    
    def F(self, enei: float, *args, **kwargs) -> Any:
        """Surface derivative of Green function."""
        return self.parent.eval(self.i, self.j, 'F', enei)
    
    def H1(self, enei: float, *args, **kwargs) -> Any:
        """F + 2*pi."""
        return self.parent.eval(self.i, self.j, 'H1', enei)
    
    def H2(self, enei: float, *args, **kwargs) -> Any:
        """F - 2*pi."""
        return self.parent.eval(self.i, self.j, 'H2', enei)


# Module-level function for compatibility with MATLAB-style function calls
def compgreenretlayer(*args, **kwargs) -> CompGreenRetLayer:
    """
    Create CompGreenRetLayer object - for compatibility with existing code.
    """
    return CompGreenRetLayer(*args, **kwargs)