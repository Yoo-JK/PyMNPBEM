"""
acagreen - Green function calculations for MNPBEM
Converted from MATLAB MEX acagreen folder

This module implements:
- Static and retarded Green functions (acagreen.cpp/h)
- Tabulated Green function interpolation (greentab.cpp/h) 
- 2D/3D interpolation utilities (interp.cpp/h)
- Particle boundary representation (particle.h)
"""

import numpy as np
from typing import Tuple, Optional, Union, Dict, Any
from abc import ABC, abstractmethod
import warnings
from scipy.interpolate import RegularGridInterpolator

# Global tree object (would be set externally)
tree = None

class Mask:
    """Cluster size and index information."""
    def __init__(self, row_size, col_size, rbegin=0, cbegin=0):
        self.row_size = row_size
        self.col_size = col_size
        self.rbegin = rbegin
        self.cbegin = cbegin
    
    def nrows(self):
        return self.row_size
    
    def ncols(self):
        return self.col_size

class Particle:
    """Discretized particle boundary."""
    def __init__(self, pos, nvec, area, z=None):
        """
        Parameters:
        -----------
        pos : np.ndarray [n, 3]
            Centroids of boundary elements
        nvec : np.ndarray [n, 3] 
            Normal vectors
        area : np.ndarray [n]
            Areas of boundary elements
        z : np.ndarray [n], optional
            Distance to closest layer (for layer structures)
        """
        self.pos = np.asarray(pos)
        self.nvec = np.asarray(nvec) 
        self.area = np.asarray(area)
        self.z = np.asarray(z) if z is not None else None
        self.n = len(area)
        
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary (MATLAB struct equivalent)."""
        return cls(
            pos=data['pos'],
            nvec=data['nvec'], 
            area=data['area'],
            z=data.get('z', None)
        )

class ACAFunc(ABC):
    """Base class for ACA function functors."""
    
    @abstractmethod
    def nrows(self) -> int:
        pass
    
    @abstractmethod
    def ncols(self) -> int:
        pass
    
    @abstractmethod
    def max_rank(self) -> int:
        pass
    
    @abstractmethod
    def getrow(self, r: int) -> np.ndarray:
        pass
    
    @abstractmethod
    def getcol(self, c: int) -> np.ndarray:
        pass

class GreenStat(ACAFunc):
    """Static Green function (real-valued)."""
    
    def __init__(self, p: Particle = None, flag: str = "G"):
        self.p = p
        self.flag = flag
        self.row = 0
        self.col = 0
        self.siz = None
        
    def __copy__(self):
        new_obj = GreenStat()
        new_obj.p = self.p
        new_obj.flag = self.flag
        return new_obj
    
    def nrows(self) -> int:
        return self.siz.nrows() if self.siz else 0
    
    def ncols(self) -> int:
        return self.siz.ncols() if self.siz else 0
    
    def max_rank(self) -> int:
        return min(self.nrows(), self.ncols())
    
    def init(self, row_cluster: int, col_cluster: int):
        """Initialize cluster."""
        global tree
        self.row = row_cluster
        self.col = col_cluster
        if tree:
            self.siz = Mask(tree.size(row_cluster), tree.size(col_cluster))
        else:
            # Fallback for testing
            self.siz = Mask(self.p.n if self.p else 1, self.p.n if self.p else 1)
    
    def getrow(self, r: int) -> np.ndarray:
        """Get row r of static Green function matrix."""
        n = self.ncols()
        b = np.zeros(n, dtype=np.float64)
        
        rr = self.siz.rbegin + r
        cc = self.siz.cbegin
        
        for c in range(n):
            # Relative position vector
            pos = self.p.pos[rr] - self.p.pos[cc + c]
            
            # Distance
            d = np.linalg.norm(pos)
            if d < 1e-12:
                d = 1e-12
            
            if self.flag == "G":
                # Green function
                b[c] = (1.0 / d) * self.p.area[cc + c]
            else:
                # Surface derivative
                dot_product = np.dot(pos, self.p.nvec[rr])
                b[c] = -dot_product / (d**3) * self.p.area[cc + c]
        
        return b
    
    def getcol(self, c: int) -> np.ndarray:
        """Get column c of static Green function matrix."""
        m = self.nrows()
        a = np.zeros(m, dtype=np.float64)
        
        rr = self.siz.rbegin
        cc = self.siz.cbegin + c
        
        for r in range(m):
            # Relative position vector
            pos = self.p.pos[rr + r] - self.p.pos[cc]
            
            # Distance
            d = np.linalg.norm(pos)
            if d < 1e-12:
                d = 1e-12
                
            if self.flag == "G":
                # Green function
                a[r] = (1.0 / d) * self.p.area[cc]
            else:
                # Surface derivative
                dot_product = np.dot(pos, self.p.nvec[rr + r])
                a[r] = -dot_product / (d**3) * self.p.area[cc]
        
        return a
    
    def eval(self, tol: float = 1e-6) -> Dict:
        """Evaluate Green function using ACA."""
        # This would require actual ACA and HMatrix implementation
        # For now, return structure that can be implemented later
        return {
            'type': 'static_green',
            'tolerance': tol,
            'particle': self.p,
            'flag': self.flag
        }

class GreenRet(ACAFunc):
    """Retarded Green function (complex-valued)."""
    
    def __init__(self, p: Particle = None, flag: str = "G", wav: complex = 1.0):
        self.p = p
        self.flag = flag
        self.wav = complex(wav)
        self.row = 0
        self.col = 0
        self.siz = None
        
    def __copy__(self):
        new_obj = GreenRet()
        new_obj.p = self.p
        new_obj.flag = self.flag
        new_obj.wav = self.wav
        return new_obj
    
    def nrows(self) -> int:
        return self.siz.nrows() if self.siz else 0
    
    def ncols(self) -> int:
        return self.siz.ncols() if self.siz else 0
    
    def max_rank(self) -> int:
        return min(self.nrows(), self.ncols())
    
    def init(self, row_cluster: int, col_cluster: int):
        """Initialize cluster."""
        global tree
        self.row = row_cluster  
        self.col = col_cluster
        if tree:
            self.siz = Mask(tree.size(row_cluster), tree.size(col_cluster))
        else:
            self.siz = Mask(self.p.n if self.p else 1, self.p.n if self.p else 1)
    
    def getrow(self, r: int) -> np.ndarray:
        """Get row r of retarded Green function matrix."""
        n = self.ncols()
        b = np.zeros(n, dtype=np.complex128)
        
        rr = self.siz.rbegin + r
        cc = self.siz.cbegin
        
        for c in range(n):
            # Relative position vector
            pos = self.p.pos[rr] - self.p.pos[cc + c]
            
            # Distance and phase factor
            d = np.linalg.norm(pos)
            if d < 1e-12:
                d = 1e-12
            fac = np.exp(1j * self.wav * d)
            
            if self.flag == "G":
                # Retarded Green function
                b[c] = (fac / d) * self.p.area[cc + c]
            else:
                # Surface derivative
                dot_product = np.dot(pos, self.p.nvec[rr])
                derivative_factor = (1j * self.wav - 1.0 / d)
                b[c] = dot_product * derivative_factor * fac / (d**2) * self.p.area[cc + c]
        
        return b
    
    def getcol(self, c: int) -> np.ndarray:
        """Get column c of retarded Green function matrix."""
        m = self.nrows()
        a = np.zeros(m, dtype=np.complex128)
        
        rr = self.siz.rbegin
        cc = self.siz.cbegin + c
        
        for r in range(m):
            # Relative position vector
            pos = self.p.pos[rr + r] - self.p.pos[cc]
            
            # Distance and phase factor
            d = np.linalg.norm(pos)
            if d < 1e-12:
                d = 1e-12
            fac = np.exp(1j * self.wav * d)
            
            if self.flag == "G":
                # Retarded Green function
                a[r] = (fac / d) * self.p.area[cc]
            else:
                # Surface derivative
                dot_product = np.dot(pos, self.p.nvec[rr + r])
                derivative_factor = (1j * self.wav - 1.0 / d)
                a[r] = dot_product * derivative_factor * fac / (d**2) * self.p.area[cc]
        
        return a
    
    def eval(self, i: int, j: int, tol: float = 1e-6) -> Dict:
        """Evaluate retarded Green function for particle pair (i,j)."""
        return {
            'type': 'retarded_green',
            'tolerance': tol,
            'particle_i': i,
            'particle_j': j,
            'particle': self.p,
            'flag': self.flag,
            'wavenumber': self.wav
        }

# Interpolation utilities
def linind(xtab: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation indices."""
    ntab = xtab.size
    h = (xtab[-1] - xtab[0]) / (ntab - 1)
    ind = np.clip(((x - xtab[0]) / h).astype(int), 0, ntab - 2)
    return ind

def logind(xtab: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Logarithmic interpolation indices."""
    ntab = xtab.size
    a, b = np.log10(xtab[0]), np.log10(xtab[-1])
    h = (b - a) / (ntab - 1)
    ind = np.clip(((np.log10(x) - a) / h).astype(int), 0, ntab - 2)
    return ind

class Interp2:
    """2D interpolation class."""
    
    def __init__(self, xtab, xflag, ytab, yflag, vtab):
        self.xtab = np.asarray(xtab)
        self.ytab = np.asarray(ytab)
        self.vtab = np.asarray(vtab)
        self.funx = linind if xflag == "lin" else logind
        self.funy = linind if yflag == "lin" else logind
    
    def __call__(self, x, y):
        """Perform 2D interpolation."""
        x, y = np.asarray(x), np.asarray(y)
        ix = self.funx(self.xtab, x)
        iy = self.funy(self.ytab, y)
        
        # Interpolation weights
        xbin = (x - self.xtab[ix]) / (self.xtab[ix + 1] - self.xtab[ix])
        ybin = (y - self.ytab[iy]) / (self.ytab[iy + 1] - self.ytab[iy])
        
        # Bilinear interpolation
        xa, xb = 1 - xbin, xbin
        ya, yb = 1 - ybin, ybin
        
        v = (xa * ya * self.vtab[ix, iy] + 
             xb * ya * self.vtab[ix + 1, iy] +
             xa * yb * self.vtab[ix, iy + 1] +
             xb * yb * self.vtab[ix + 1, iy + 1])
        
        return v

class Interp3:
    """3D interpolation class."""
    
    def __init__(self, xtab, xflag, ytab, yflag, ztab, zflag, vtab):
        self.xtab = np.asarray(xtab)
        self.ytab = np.asarray(ytab)
        self.ztab = np.asarray(ztab)
        self.vtab = np.asarray(vtab)
        self.funx = linind if xflag == "lin" else logind
        self.funy = linind if yflag == "lin" else logind
        self.funz = linind if zflag == "lin" else logind
    
    def __call__(self, x, y, z):
        """Perform 3D interpolation."""
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        ix = self.funx(self.xtab, x)
        iy = self.funy(self.ytab, y)
        iz = self.funz(self.ztab, z)
        
        # Interpolation weights
        xbin = (x - self.xtab[ix]) / (self.xtab[ix + 1] - self.xtab[ix])
        ybin = (y - self.ytab[iy]) / (self.ytab[iy + 1] - self.ytab[iy])
        zbin = (z - self.ztab[iz]) / (self.ztab[iz + 1] - self.ztab[iz])
        
        # Trilinear interpolation
        xa, xb = 1 - xbin, xbin
        ya, yb = 1 - ybin, ybin
        za, zb = 1 - zbin, zbin
        
        v = (xa * ya * za * self.vtab[ix, iy, iz] +
             xb * ya * za * self.vtab[ix + 1, iy, iz] +
             xa * yb * za * self.vtab[ix, iy + 1, iz] +
             xb * yb * za * self.vtab[ix + 1, iy + 1, iz] +
             xa * ya * zb * self.vtab[ix, iy, iz + 1] +
             xb * ya * zb * self.vtab[ix + 1, iy, iz + 1] +
             xa * yb * zb * self.vtab[ix, iy + 1, iz + 1] +
             xb * yb * zb * self.vtab[ix + 1, iy + 1, iz + 1])
        
        return v

# Tabulated Green functions base class
class GreenTab(ACAFunc):
    """Base class for tabulated Green functions."""
    
    def __init__(self, p: Particle = None):
        self.p = p
        self.row = 0
        self.col = 0
        self.siz = None
        
    def nrows(self) -> int:
        return self.siz.nrows() if self.siz else 0
    
    def ncols(self) -> int:
        return self.siz.ncols() if self.siz else 0
    
    def max_rank(self) -> int:
        return min(self.nrows(), self.ncols())
    
    def init(self, row_cluster: int, col_cluster: int):
        """Initialize cluster."""
        global tree
        self.row = row_cluster
        self.col = col_cluster
        if tree:
            self.siz = Mask(tree.size(row_cluster), tree.size(col_cluster))
        else:
            self.siz = Mask(self.p.n if self.p else 1, self.p.n if self.p else 1)
    
    def eval(self, i: int, j: int, tol: float = 1e-6) -> Dict:
        """Evaluate tabulated Green function."""
        return {
            'type': 'tabulated_green',
            'tolerance': tol,
            'particle_i': i,
            'particle_j': j
        }

class GreenTabG2(GreenTab):
    """2D interpolation of Green function."""
    
    def __init__(self, p: Particle, tab_data: Dict, layer_index: int):
        super().__init__(p)
        self.gtab = Interp2(
            tab_data['r'], tab_data['rmod'],
            tab_data['z1'], tab_data['zmod'],
            tab_data['G']
        )
        self.uplo = 'U' if layer_index == 1 else 'L'
        self.rmin = tab_data['r'][0]
    
    def getrow(self, r: int) -> np.ndarray:
        """Get row for 2D Green function."""
        n = self.ncols()
        b = np.zeros(n, dtype=np.complex128)
        
        rr = self.siz.rbegin + r
        cc = self.siz.cbegin
        
        rho = np.zeros(n)
        z = np.zeros(n)
        
        for c in range(n):
            # Polar distance
            dx = self.p.pos[rr, 0] - self.p.pos[cc + c, 0]
            dy = self.p.pos[rr, 1] - self.p.pos[cc + c, 1]
            rho[c] = max(self.rmin, np.sqrt(dx**2 + dy**2))
            
            # Z-distance
            z[c] = self.p.z[rr] + (1 if self.uplo == 'U' else -1) * self.p.z[cc + c]
        
        # Interpolation
        g = self.gtab(rho, z)
        
        # Green function
        for c in range(n):
            d = np.sqrt(rho[c]**2 + z[c]**2)
            b[c] = g[c] / d * self.p.area[cc + c]
        
        return b
    
    def getcol(self, c: int) -> np.ndarray:
        """Get column for 2D Green function."""
        m = self.nrows()
        a = np.zeros(m, dtype=np.complex128)
        
        rr = self.siz.rbegin
        cc = self.siz.cbegin + c
        
        rho = np.zeros(m)
        z = np.zeros(m)
        
        for r in range(m):
            # Polar distance
            dx = self.p.pos[rr + r, 0] - self.p.pos[cc, 0]
            dy = self.p.pos[rr + r, 1] - self.p.pos[cc, 1]
            rho[r] = max(self.rmin, np.sqrt(dx**2 + dy**2))
            
            # Z-distance
            z[r] = self.p.z[rr + r] + (1 if self.uplo == 'U' else -1) * self.p.z[cc]
        
        # Interpolation
        g = self.gtab(rho, z)
        
        # Green function
        for r in range(m):
            d = np.sqrt(rho[r]**2 + z[r]**2)
            a[r] = g[r] / d * self.p.area[cc]
        
        return a

# Convenience functions
def green_static(p: Particle, flag: str = "G") -> GreenStat:
    """Create static Green function."""
    return GreenStat(p, flag)

def green_retarded(p: Particle, wav: complex, flag: str = "G") -> GreenRet:
    """Create retarded Green function."""
    return GreenRet(p, flag, wav)