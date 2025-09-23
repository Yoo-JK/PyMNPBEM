import numpy as np
from .bembase import BemBase
from .compgreenret import CompGreenRet
from .compstructmirror import CompStructMirror
from .compstruct import CompStruct


class CompGreenRetMirror(BemBase):
    """Green function for composite particles with mirror symmetry."""
    
    # Class attributes
    name = 'greenfunction'
    needs = [{'sim': 'ret'}, 'sym']
    
    def __init__(self, p, dummy=None, *args, **kwargs):
        """
        Initialize Green functions for composite object & mirror symmetry.
        
        Parameters
        ----------
        p : object
            Green function for particle p with mirror symmetry
        dummy : None
            Dummy input to have same calling sequence as compgreen
        *args, **kwargs : optional
            Additional options (see green/options)
        """
        super().__init__()
        self.p = p
        # Initialize Green function
        self.g = CompGreenRet(p, p.full(), *args, **kwargs)
    
    def __str__(self):
        """Command window display."""
        info = {
            'p': self.p,
            'g': self.g
        }
        return f"CompGreenRetMirror:\n{info}"
    
    def eval(self, *args, **kwargs):
        """
        Evaluate retarded Green function with mirror symmetry.
        
        Parameters
        ----------
        *args, **kwargs : various
            Keys like 'G', 'F', 'H1', 'H2', 'Gp', 'H1p', 'H2p' and energy
            
        Returns
        -------
        list
            Green function matrices with symmetry applied
        """
        # Pass input to CompGreenRet
        mat = self.g.__getitem__(*args, **kwargs)
        
        # Symmetry table
        tab = self.p.symtable
        
        # Allocate output array
        g = [np.zeros_like(mat) for _ in range(tab.shape[0])]
        
        if min(mat.shape) != 1:
            # Size of Green function matrix
            siz = mat.shape
            
            # Decompose Green matrix into sub-matrices
            if len(siz) == 2:
                # G, F, H1, H2
                n_blocks = siz[1] // siz[0]
                mat_blocks = np.split(mat, n_blocks, axis=1)
            else:
                # Gp, H1p, H2p
                n_blocks = siz[2] // siz[0]
                mat_blocks = np.split(mat, n_blocks, axis=2)
            
            # Contract Green function for different symmetry values
            for i in range(tab.shape[0]):
                for j in range(tab.shape[1]):
                    g[i] = g[i] + tab[i, j] * mat_blocks[j]
        
        return g
    
    def field(self, sig, inout=1):
        """
        Electric and magnetic field inside/outside of particle surface.
        
        Parameters
        ----------
        sig : object
            Surface charges and currents
        inout : int, optional
            Fields inside (inout=1, default) or outside (inout=2) of particle surface
            
        Returns
        -------
        CompStructMirror
            Electric and magnetic fields
        """
        # Cannot compute fields from just normal surface derivative
        assert self.g.deriv == 'cart', "Cartesian derivatives required"
        
        # Wavelength and wavenumber of light in vacuum
        enei = sig.enei
        k = 2 * np.pi / sig.enei
        
        # Allocate output
        field = CompStructMirror(sig.p, sig.enei, sig.fun)
        
        # Green function
        G1 = self.__getitem__(inout, 1).G(enei)
        G2 = self.__getitem__(inout, 2).G(enei)
        
        # Derivative of Green function
        if inout == 1:
            H1p = self.__getitem__(inout, 1).H1p(enei)
            H2p = self.__getitem__(inout, 2).H1p(enei)
        else:
            H1p = self.__getitem__(inout, 1).H2p(enei)
            H2p = self.__getitem__(inout, 2).H2p(enei)
        
        # Loop over symmetry values
        for i in range(len(sig.val)):
            # Surface charge
            isig = sig.val[i]
            
            # Index of symmetry values within symmetry table
            x = self.p.symindex(isig.symval[0, :])
            y = self.p.symindex(isig.symval[1, :])
            z = self.p.symindex(isig.symval[2, :])
            
            # Index array
            ind = [x, y, z]
            
            # Electric field E = i k A - grad V
            e = (1j * k * self._indmul(G1, isig.h1, ind) - 
                 np.matmul(H1p[z], isig.sig1) +
                 1j * k * self._indmul(G2, isig.h2, ind) - 
                 np.matmul(H2p[z], isig.sig2))
            
            # Magnetic field
            h = (self._indcross(H1p, isig.h1, ind) + 
                 self._indcross(H2p, isig.h2, ind))
            
            # Set output
            field.val[i] = CompStruct(sig.p, sig.enei, e=e, h=h)
            # Set symmetry value
            field.val[i].symval = sig.val[i].symval
        
        return field
    
    def _indmul(self, mat, v, ind):
        """Indexed matrix multiplication."""
        if len(mat) == 1 and mat[0] == 0:
            return 0
        else:
            siz = list(v.shape)
            siz[1] = 1
            
            result = np.concatenate([
                np.matmul(mat[ind[0]], v[:, 0, :].reshape(siz)),
                np.matmul(mat[ind[1]], v[:, 1, :].reshape(siz)),
                np.matmul(mat[ind[2]], v[:, 2, :].reshape(siz))
            ], axis=1)
            
            return result
    
    def _indcross(self, mat, v, ind):
        """Indexed cross product."""
        if len(mat) == 1 and mat[0] == 0:
            return 0
        else:
            siz = list(v.shape)
            siz[1] = 1
            
            # Matrix and vector function
            def imat(k, i):
                return np.squeeze(mat[ind[k]][:, i, :])
            
            def ivec(i):
                return v[:, i, :].reshape(siz)
            
            # Cross product components
            u1 = np.matmul(imat(2, 1), ivec(2)) - np.matmul(imat(1, 2), ivec(1))
            u2 = np.matmul(imat(0, 2), ivec(0)) - np.matmul(imat(2, 0), ivec(2))
            u3 = np.matmul(imat(1, 0), ivec(1)) - np.matmul(imat(0, 1), ivec(0))
            
            return np.concatenate([u1, u2, u3], axis=1)
    
    def potential(self, sig, inout=1):
        """
        Determine potentials and surface derivatives inside/outside of particle.
        Computed from solutions of full Maxwell equations.
        
        Parameters
        ----------
        sig : object
            Surface charges
        inout : int, optional
            Potentials inside (inout=1, default) or outside (inout=2) of particle
            
        Returns
        -------
        CompStructMirror
            Potentials and surface derivatives
        """
        # Wavelength of light in vacuum
        enei = sig.enei
        
        # Allocate output
        pot = CompStructMirror(sig.p, sig.enei, sig.fun)
        
        # Inside or outside surface derivative
        H_key = 'H1' if inout == 1 else 'H2'
        
        # Green function
        G1 = self.__getitem__(inout, 1).G(enei)
        G2 = self.__getitem__(inout, 2).G(enei)
        
        # Surface derivatives of Green function
        H1 = getattr(self.__getitem__(inout, 1), H_key)(enei)
        H2 = getattr(self.__getitem__(inout, 2), H_key)(enei)
        
        # Loop over symmetry values
        for i in range(len(sig.val)):
            # Surface charge
            isig = sig.val[i]
            
            # Index of symmetry values within symmetry table
            x = self.p.symindex(isig.symval[0, :])
            y = self.p.symindex(isig.symval[1, :])
            z = self.p.symindex(isig.symval[2, :])
            
            # Index array
            ind = [x, y, z]
            
            # Scalar potential
            phi = np.matmul(G1[z], isig.sig1) + np.matmul(G2[z], isig.sig2)
            phip = np.matmul(H1[z], isig.sig1) + np.matmul(H2[z], isig.sig2)
            
            # Vector potential
            a = self._indmul_pot(G1, isig.h1, ind) + self._indmul_pot(G2, isig.h2, ind)
            ap = self._indmul_pot(H1, isig.h1, ind) + self._indmul_pot(H2, isig.h2, ind)
            
            if inout == 1:
                pot.val[i] = CompStruct(sig.p, enei, 
                                      phi1=phi, phi1p=phip, a1=a, a1p=ap)
            else:
                pot.val[i] = CompStruct(sig.p, enei, 
                                      phi2=phi, phi2p=phip, a2=a, a2p=ap)
            
            # Set symmetry value
            pot.val[i].symval = sig.val[i].symval
        
        return pot
    
    def _indmul_pot(self, mat, v, ind):
        """Matrix multiplication for given index (potential version)."""
        if len(mat) == 1 and mat[0] == 0:
            return 0
        else:
            siz = list(v.shape)
            siz[1] = 1
            
            result = np.concatenate([
                np.matmul(mat[ind[0]], v[:, 0, :].reshape(siz)),
                np.matmul(mat[ind[1]], v[:, 1, :].reshape(siz)),
                np.matmul(mat[ind[2]], v[:, 2, :].reshape(siz))
            ], axis=1)
            
            return result
    
    def __getitem__(self, *indices):
        """
        Derived properties for CompGreenRetMirror objects.
        
        Usage:
            obj[i, j].G(enei)  : composite Green function
        """
        return self.eval(*indices)
    
    def __getattr__(self, name):
        """Handle attribute access for field and potential methods."""
        if name == 'con':
            return getattr(self.g, name)
        else:
            return super().__getattribute__(name)