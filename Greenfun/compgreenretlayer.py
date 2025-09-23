import numpy as np
from .bembase import BemBase
from .compgreenret import CompGreenRet
from .compstruct import CompStruct
from .greenretlayer import GreenRetLayer
from .bemoptions import get_bemoptions


class CompGreenRetLayer(BemBase):
    """Green function for layer structure."""
    
    # Class attributes
    name = 'greenfunction'
    needs = [{'sim': 'ret'}, 'layer']
    
    def __init__(self, p1, p2, *args, **kwargs):
        """
        Initialize Green function object for layer structure.
        
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
        self.p1 = p1
        self.p2 = p2
        self.g = None
        self.gr = None
        self.layer = None
        self.ind1 = None
        self.ind2 = None
        self._init(*args, **kwargs)
    
    def __str__(self):
        """Command window display."""
        info = {
            'p1': self.p1,
            'p2': self.p2,
            'g': self.g,
            'gr': self.gr
        }
        return f"CompGreenRetLayer:\n{info}"
    
    def _init(self, *args, **kwargs):
        """Initialize Green function object and layer structure."""
        # Options for Green function
        op = get_bemoptions(['greenlayer', 'greenretlayer'], *args, **kwargs)
        
        # Initialize Green function and layer structure
        self.g = CompGreenRet(self.p1, self.p2, *args, **kwargs)
        self.layer = op.layer
        
        # Inout argument
        inout1 = self.p1.expand([self.p1.inout[:, -1]])
        inout2 = self.p2.expand([self.p2.inout[:, -1]])
        
        # Find elements connected to layer structure
        self.ind1 = np.where(np.any(inout1[:, None] == self.layer.ind, axis=1))[0]
        self.ind2 = np.where(np.any(inout2[:, None] == self.layer.ind, axis=1))[0]
        
        # Compoint and comparticle objects with boundary elements connected to layer structure
        p1_combined = np.vstack([p for p in self.p1.p])
        p2_combined = np.vstack([p for p in self.p2.p])
        
        p1_selected = p1_combined.select(index=self.ind1)
        p2_selected = p2_combined.select(index=self.ind2)
        
        # Initialize reflected part of Green function
        self.gr = GreenRetLayer(p1_selected, p2_selected, *args, **kwargs)
    
    def eval(self, *args, **kwargs):
        """
        Evaluate Green function.
        
        Parameters
        ----------
        *args : various
            i1, i2, enei, key for full matrix or i1, i2, enei, key, ind for selected elements
            
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
    
    def _eval1(self, i1, i2, key, enei):
        """Evaluate Green function for full matrix."""
        # Compute Green functions
        g = self.g.eval(i1, i2, key, enei)
        
        # Make sure that g is not zero
        if np.isscalar(g) and g == 0:
            if key in ['Gp', 'H1p', 'H2p']:
                g = np.zeros((self.p1.n, 3, self.p2.n))
            else:
                g = np.zeros((self.p1.n, self.p2.n))
        
        # Green function for outer surfaces
        if i1 == self.p1.inout.shape[1] and i2 == 2:
            # Initialize reflected part of Green function
            self._initrefl(enei)
            
            # Add reflected Green function
            if key == 'G':
                g = self._assembly_full(g, self.gr.G)
            elif key in ['F', 'H1', 'H2']:
                g = self._assembly_full(g, self.gr.F)
            elif key in ['Gp', 'H1p', 'H2p']:
                g = self._assembly_full(g, self.gr.Gp)
        
        return g
    
    def _eval2(self, i1, i2, key, enei, ind):
        """Evaluate Green function for selected matrix elements."""
        # Compute Green functions
        g = self.g.eval(i1, i2, key, enei, ind)
        
        # Make sure that g is not zero
        if np.isscalar(g) and g == 0:
            g = np.zeros((self.p1.n, self.p2.n))
        
        # Green function for outer surfaces
        if i1 == self.p1.inout.shape[1] and i2 == 2:
            # Rows and columns of selected matrix elements
            row, col = np.unravel_index(ind, (self.p1.n, self.p2.n))
            
            # Find elements connected to layer structure
            i1_mask = np.isin(row, self.ind1)
            i2_mask = np.isin(col, self.ind2)
            ilayer = i1_mask & i2_mask
            
            i1_indices = np.searchsorted(self.ind1, row[ilayer])
            i2_indices = np.searchsorted(self.ind2, col[ilayer])
            
            # Convert subscripts to linear indices
            ind1 = np.ravel_multi_index((i1_indices, i2_indices), 
                                       (self.gr.p1.n, self.gr.p2.n))
            
            # Initialize reflected part of Green function
            self._initrefl(enei, ind1)
            
            # Add reflected Green function
            if key == 'G':
                g = self._assembly_selected(g, self.gr.G, ilayer)
            elif key in ['F', 'H1', 'H2']:
                g = self._assembly_selected(g, self.gr.F, ilayer)
        
        return g
    
    def _assembly_full(self, g, gr):
        """Assemble free and reflected Green functions for full matrix."""
        G = {}
        names = gr.keys() if isinstance(gr, dict) else ['ss', 'hh', 'p']
        
        for name in names:
            if name in ['ss', 'hh', 'p']:
                G[name] = g.copy()
            else:
                G[name] = np.zeros_like(g)
            
            if len(g.shape) == 2:
                G[name][np.ix_(self.ind1, self.ind2)] += gr[name]
            else:
                G[name][np.ix_(self.ind1, slice(None), self.ind2)] += gr[name]
        
        return G
    
    def _assembly_selected(self, g, gr, ind):
        """Assemble free and reflected Green functions for selected elements."""
        G = {}
        names = gr.keys() if isinstance(gr, dict) else ['ss', 'hh', 'p']
        
        for name in names:
            if name in ['ss', 'hh', 'p']:
                G[name] = g.copy()
            else:
                G[name] = np.zeros_like(g)
            
            G[name][ind] += gr[name]
        
        return G
    
    def _initrefl(self, *args, **kwargs):
        """Initialize reflected part of Green function."""
        self.gr = self.gr.initrefl(*args, **kwargs)
    
    def field(self, sig, inout=1):
        """
        Electric and magnetic fields inside/outside of particle surface.
        
        Parameters
        ----------
        sig : CompStruct
            Surface charges and currents
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
        
        # Initialize reflected Green functions
        self._initrefl(enei)
        
        # Green function and E = i k A
        e = (1j * k * (self._matmul2(self.eval(inout, 1, 'G', enei), sig, 'h1') +
                      self._matmul2(self.eval(inout, 2, 'G', enei), sig, 'h2')))
        
        # Derivative of Green function
        if inout == 1:
            H1p = self.eval(inout, 1, 'H1p', enei)
            H2p = self.eval(inout, 2, 'H1p', enei)
        else:
            H1p = self.eval(inout, 1, 'H2p', enei)
            H2p = self.eval(inout, 2, 'H2p', enei)
        
        # Add derivative of scalar potential to electric field
        e = e - self._matmul2(H1p, sig, 'sig1') - self._matmul2(H2p, sig, 'sig2')
        
        # Magnetic field
        h = self._cross(H1p, sig, 'h1') + self._cross(H2p, sig, 'h2')
        
        # Set output
        return CompStruct(self.p1, enei, e=e, h=h)
    
    def _cross(self, G, sig, name):
        """Multidimensional cross product."""
        if np.isscalar(G) and not isinstance(G, dict):
            return 0
        
        # Cross product
        cross = np.concatenate([
            self._matmul3(G, sig, name, 1, 2) - self._matmul3(G, sig, name, 2, 1),
            self._matmul3(G, sig, name, 2, 0) - self._matmul3(G, sig, name, 0, 2),
            self._matmul3(G, sig, name, 0, 1) - self._matmul3(G, sig, name, 1, 0)
        ], axis=1)
        
        return cross
    
    def potential(self, sig, inout=1):
        """
        Potentials and surface derivatives inside/outside of particle.
        
        Parameters
        ----------
        sig : CompStruct
            Surface charges (see bemstat)
        inout : int, optional
            Potentials inside (inout=1, default) or outside (inout=2) of particle
            
        Returns
        -------
        CompStruct
            Potentials and surface derivatives
        """
        enei = sig.enei
        
        # Initialize reflected Green functions
        self._initrefl(enei)
        
        # Set parameters that depend on inside/outside
        H_key = 'H1' if inout == 1 else 'H2'
        
        # Green functions
        G1 = self.__getitem__(inout, 1).G(enei)
        G2 = self.__getitem__(inout, 2).G(enei)
        
        # Surface derivatives of Green functions
        H1 = getattr(self.__getitem__(inout, 1), H_key)(enei)
        H2 = getattr(self.__getitem__(inout, 2), H_key)(enei)
        
        # Potential and surface derivative
        # Scalar potential
        phi = self._matmul2(G1, sig, 'sig1') + self._matmul2(G2, sig, 'sig2')
        phip = self._matmul2(H1, sig, 'sig1') + self._matmul2(H2, sig, 'sig2')
        
        # Vector potential
        a = self._matmul2(G1, sig, 'h1') + self._matmul2(G2, sig, 'h2')
        ap = self._matmul2(H1, sig, 'h1') + self._matmul2(H2, sig, 'h2')
        
        # Set output
        if inout == 1:
            return CompStruct(self.p1, enei, phi1=phi, phi1p=phip, a1=a, a1p=ap)
        else:
            return CompStruct(self.p1, enei, phi2=phi, phi2p=phip, a2=a, a2p=ap)
    
    def _matmul2(self, G, sig, name):
        """Matrix multiplication."""
        if not isinstance(G, dict):
            return np.matmul(G, getattr(sig, name))
        else:
            if name == 'sig1':
                # G.ss * sig1 + G.sh * h1[2]
                return (np.matmul(G['ss'], sig.sig1) +
                       np.matmul(G['sh'], sig.h1[:, 2, :].reshape(sig.sig1.shape)))
            elif name == 'sig2':
                # G.ss * sig2 + G.sh * h2[2]
                return (np.matmul(G['ss'], sig.sig2) +
                       np.matmul(G['sh'], sig.h2[:, 2, :].reshape(sig.sig2.shape)))
            elif name == 'h1':
                siz = list(sig.h1.shape)
                siz[1] = 1
                return np.concatenate([
                    np.matmul(G['p'], sig.h1[:, 0, :].reshape(siz)),
                    np.matmul(G['p'], sig.h1[:, 1, :].reshape(siz)),
                    (np.matmul(G['hh'], sig.h1[:, 2, :].reshape(siz)) +
                     np.matmul(G['hs'], sig.sig1.reshape(siz)))
                ], axis=1)
            elif name == 'h2':
                siz = list(sig.h2.shape)
                siz[1] = 1
                return np.concatenate([
                    np.matmul(G['p'], sig.h2[:, 0, :].reshape(siz)),
                    np.matmul(G['p'], sig.h2[:, 1, :].reshape(siz)),
                    (np.matmul(G['hh'], sig.h2[:, 2, :].reshape(siz)) +
                     np.matmul(G['hs'], sig.sig2.reshape(siz)))
                ], axis=1)
    
    def _matmul3(self, G, sig, name, i1, i2):
        """Matrix multiplication for cross product."""
        if not isinstance(G, dict):
            # Size of output matrix
            siz = list(sig.h1.shape)
            siz[0:2] = [G.shape[0], 1]
            # G[i1] * h[i2]
            return np.matmul(G[:, i1, :], getattr(sig, name)[:, i2, :]).reshape(siz)
        else:
            # Size of output matrix
            siz = list(sig.h1.shape)
            siz[0:2] = [G['p'].shape[0], 1]
            
            # Surface charge and current
            if name == 'h1':
                sig_val, h = sig.sig1, sig.h1
            elif name == 'h2':
                sig_val, h = sig.sig2, sig.h2
            
            # Treat parallel and perpendicular components differently
            if i2 in [0, 1]:
                return np.matmul(G['p'][:, i1, :], h[:, i2, :]).reshape(siz)
            elif i2 == 2:
                return (np.matmul(G['hh'][:, i1, :], h[:, i2, :]).reshape(siz) +
                       np.matmul(G['hs'][:, i1, :], sig_val).reshape(siz))
    
    def __getitem__(self, *indices):
        """
        Derived properties for CompGreenRetLayer objects.
        
        Usage:
            obj[i, j].G(enei) : composite Green function
        """
        return self.eval(*indices)
    
    def __call__(self, *args, **kwargs):
        """Initialize reflected part of Green function."""
        return self._initrefl(*args, **kwargs)
    
    def __getattr__(self, name):
        """Handle attribute access."""
        if name == 'deriv':
            return self.gr.deriv
        else:
            return super().__getattribute__(name)