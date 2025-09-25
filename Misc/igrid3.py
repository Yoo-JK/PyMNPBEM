import numpy as np
from typing import Tuple, Callable, Union, Any

class IGrid3:
    """
    3D grid for interpolation using trilinear interpolation.
    
    This class provides functionality for trilinear interpolation on a 3D grid,
    including computation of derivatives along specified directions.
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """
        Initialize 3D grid for interpolation.
        
        Parameters:
        -----------
        x : array_like
            x-values of grid points
        y : array_like  
            y-values of grid points
        z : array_like
            z-values of grid points
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.z = np.asarray(z)
        
        # Ensure arrays are sorted for proper interpolation
        if not np.all(np.diff(self.x) > 0):
            raise ValueError("x values must be strictly increasing")
        if not np.all(np.diff(self.y) > 0):
            raise ValueError("y values must be strictly increasing")
        if not np.all(np.diff(self.z) > 0):
            raise ValueError("z values must be strictly increasing")
    
    @property
    def size(self) -> Tuple[int, int, int]:
        """Size of grid used for interpolation."""
        return (len(self.x), len(self.y), len(self.z))
    
    @property
    def numel(self) -> int:
        """Number of grid elements."""
        return len(self.x) * len(self.y) * len(self.z)
    
    def finterp(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create interpolation function for points (x,y,z).
        
        Parameters:
        -----------
        x, y, z : array_like
            Interpolation points
            
        Returns:
        --------
        callable
            Interpolation function that takes array V with tabulated values
            and returns interpolated values at positions (x,y,z)
        """
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        
        # Find indices for grid positions using searchsorted (equivalent to histc)
        ix = np.searchsorted(self.x, x_flat, side='right') - 1
        iy = np.searchsorted(self.y, y_flat, side='right') - 1
        iz = np.searchsorted(self.z, z_flat, side='right') - 1
        
        # Check bounds
        if np.any(ix < 0) or np.any(ix >= len(self.x) - 1):
            raise ValueError("x values are outside interpolation range")
        if np.any(iy < 0) or np.any(iy >= len(self.y) - 1):
            raise ValueError("y values are outside interpolation range")
        if np.any(iz < 0) or np.any(iz >= len(self.z) - 1):
            raise ValueError("z values are outside interpolation range")
        
        # Handle edge cases where coordinates equal maximum values
        ix = np.minimum(ix, len(self.x) - 2)
        iy = np.minimum(iy, len(self.y) - 2)
        iz = np.minimum(iz, len(self.z) - 2)
        
        # Compute normalized coordinates within each grid cell
        xx = (x_flat - self.x[ix]) / (self.x[ix + 1] - self.x[ix])
        yy = (y_flat - self.y[iy]) / (self.y[iy + 1] - self.y[iy])
        zz = (z_flat - self.z[iz]) / (self.z[iz + 1] - self.z[iz])
        
        # Grid size for computing linear indices
        siz = (len(self.x), len(self.y), len(self.z))
        
        # Compute linear indices for the eight corners of each grid cell
        # Order: (i,j,k), (i+1,j,k), (i,j+1,k), (i+1,j+1,k),
        #        (i,j,k+1), (i+1,j,k+1), (i,j+1,k+1), (i+1,j+1,k+1)
        ind = np.column_stack([
            np.ravel_multi_index((ix, iy, iz), siz),          # (i, j, k)
            np.ravel_multi_index((ix + 1, iy, iz), siz),      # (i+1, j, k)
            np.ravel_multi_index((ix, iy + 1, iz), siz),      # (i, j+1, k)
            np.ravel_multi_index((ix + 1, iy + 1, iz), siz),  # (i+1, j+1, k)
            np.ravel_multi_index((ix, iy, iz + 1), siz),      # (i, j, k+1)
            np.ravel_multi_index((ix + 1, iy, iz + 1), siz),  # (i+1, j, k+1)
            np.ravel_multi_index((ix, iy + 1, iz + 1), siz),  # (i, j+1, k+1)
            np.ravel_multi_index((ix + 1, iy + 1, iz + 1), siz)  # (i+1, j+1, k+1)
        ])
        
        # Trilinear interpolation weights
        w = np.column_stack([
            (1 - xx) * (1 - yy) * (1 - zz),  # weight for (i, j, k)
            xx * (1 - yy) * (1 - zz),        # weight for (i+1, j, k)
            (1 - xx) * yy * (1 - zz),        # weight for (i, j+1, k)
            xx * yy * (1 - zz),              # weight for (i+1, j+1, k)
            (1 - xx) * (1 - yy) * zz,        # weight for (i, j, k+1)
            xx * (1 - yy) * zz,              # weight for (i+1, j, k+1)
            (1 - xx) * yy * zz,              # weight for (i, j+1, k+1)
            xx * yy * zz                     # weight for (i+1, j+1, k+1)
        ])
        
        def interpolate(v: np.ndarray) -> np.ndarray:
            """Perform trilinear interpolation."""
            v_flat = v.flatten()
            result = np.sum(w * v_flat[ind], axis=1)
            return result.reshape(x.shape)
        
        return interpolate
    
    def fderiv(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, direction: int) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create derivative function for points (x,y,z) along specified direction.
        
        Parameters:
        -----------
        x, y, z : array_like
            Points where derivative is computed
        direction : int
            Derivative direction (1 for x, 2 for y, 3 for z)
            
        Returns:
        --------
        callable
            Function that computes derivative of interpolated values
        """
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        
        # Find indices for grid positions
        ix = np.searchsorted(self.x, x_flat, side='right') - 1
        iy = np.searchsorted(self.y, y_flat, side='right') - 1
        iz = np.searchsorted(self.z, z_flat, side='right') - 1
        
        # Check bounds
        if np.any(ix < 0) or np.any(ix >= len(self.x) - 1):
            raise ValueError("x values are outside interpolation range")
        if np.any(iy < 0) or np.any(iy >= len(self.y) - 1):
            raise ValueError("y values are outside interpolation range")
        if np.any(iz < 0) or np.any(iz >= len(self.z) - 1):
            raise ValueError("z values are outside interpolation range")
        
        # Handle edge cases
        ix = np.minimum(ix, len(self.x) - 2)
        iy = np.minimum(iy, len(self.y) - 2)
        iz = np.minimum(iz, len(self.z) - 2)
        
        # Bin sizes and normalized coordinates
        hx = self.x[ix + 1] - self.x[ix]
        hy = self.y[iy + 1] - self.y[iy]
        hz = self.z[iz + 1] - self.z[iz]
        xx = (x_flat - self.x[ix]) / hx
        yy = (y_flat - self.y[iy]) / hy
        zz = (z_flat - self.z[iz]) / hz
        
        # Grid size for computing linear indices
        siz = (len(self.x), len(self.y), len(self.z))
        
        # Compute linear indices for the eight corners
        ind = np.column_stack([
            np.ravel_multi_index((ix, iy, iz), siz),
            np.ravel_multi_index((ix + 1, iy, iz), siz),
            np.ravel_multi_index((ix, iy + 1, iz), siz),
            np.ravel_multi_index((ix + 1, iy + 1, iz), siz),
            np.ravel_multi_index((ix, iy, iz + 1), siz),
            np.ravel_multi_index((ix + 1, iy, iz + 1), siz),
            np.ravel_multi_index((ix, iy + 1, iz + 1), siz),
            np.ravel_multi_index((ix + 1, iy + 1, iz + 1), siz)
        ])
        
        # Derivative weights
        if direction == 1:  # derivative w.r.t. x
            w = np.column_stack([
                (-1 / hx) * (1 - yy) * (1 - zz),  # d/dx weight for (i, j, k)
                (1 / hx) * (1 - yy) * (1 - zz),   # d/dx weight for (i+1, j, k)
                (-1 / hx) * yy * (1 - zz),        # d/dx weight for (i, j+1, k)
                (1 / hx) * yy * (1 - zz),         # d/dx weight for (i+1, j+1, k)
                (-1 / hx) * (1 - yy) * zz,        # d/dx weight for (i, j, k+1)
                (1 / hx) * (1 - yy) * zz,         # d/dx weight for (i+1, j, k+1)
                (-1 / hx) * yy * zz,              # d/dx weight for (i, j+1, k+1)
                (1 / hx) * yy * zz                # d/dx weight for (i+1, j+1, k+1)
            ])
        elif direction == 2:  # derivative w.r.t. y
            w = np.column_stack([
                (1 - xx) * (-1 / hy) * (1 - zz),  # d/dy weight for (i, j, k)
                xx * (-1 / hy) * (1 - zz),        # d/dy weight for (i+1, j, k)
                (1 - xx) * (1 / hy) * (1 - zz),   # d/dy weight for (i, j+1, k)
                xx * (1 / hy) * (1 - zz),         # d/dy weight for (i+1, j+1, k)
                (1 - xx) * (-1 / hy) * zz,        # d/dy weight for (i, j, k+1)
                xx * (-1 / hy) * zz,              # d/dy weight for (i+1, j, k+1)
                (1 - xx) * (1 / hy) * zz,         # d/dy weight for (i, j+1, k+1)
                xx * (1 / hy) * zz                # d/dy weight for (i+1, j+1, k+1)
            ])
        elif direction == 3:  # derivative w.r.t. z
            w = np.column_stack([
                (1 - xx) * (1 - yy) * (-1 / hz),  # d/dz weight for (i, j, k)
                xx * (1 - yy) * (-1 / hz),        # d/dz weight for (i+1, j, k)
                (1 - xx) * yy * (-1 / hz),        # d/dz weight for (i, j+1, k)
                xx * yy * (-1 / hz),              # d/dz weight for (i+1, j+1, k)
                (1 - xx) * (1 - yy) * (1 / hz),   # d/dz weight for (i, j, k+1)
                xx * (1 - yy) * (1 / hz),         # d/dz weight for (i+1, j, k+1)
                (1 - xx) * yy * (1 / hz),         # d/dz weight for (i, j+1, k+1)
                xx * yy * (1 / hz)                # d/dz weight for (i+1, j+1, k+1)
            ])
        else:
            raise ValueError("direction must be 1 (x), 2 (y), or 3 (z)")
        
        def derivative(v: np.ndarray) -> np.ndarray:
            """Compute derivative of interpolated values."""
            v_flat = v.flatten()
            result = np.sum(w * v_flat[ind], axis=1)
            return result.reshape(x.shape)
        
        return derivative
    
    def __call__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Perform interpolation directly.
        
        Parameters:
        -----------
        x, y, z : array_like
            Interpolation points
        v : array_like
            Values at grid points (must have shape compatible with self.size)
            
        Returns:
        --------
        np.ndarray
            Interpolated values at positions (x, y, z)
        """
        interpolation_func = self.finterp(x, y, z)
        return interpolation_func(v)
    
    def __str__(self) -> str:
        """String representation."""
        return f"IGrid3(x: {len(self.x)} points, y: {len(self.y)} points, z: {len(self.z)} points)"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"IGrid3(\n"
                f"  x: [{self.x[0]:.3f}, ..., {self.x[-1]:.3f}] ({len(self.x)} points)\n"
                f"  y: [{self.y[0]:.3f}, ..., {self.y[-1]:.3f}] ({len(self.y)} points)\n"
                f"  z: [{self.z[0]:.3f}, ..., {self.z[-1]:.3f}] ({len(self.z)} points)\n"
                f")")


# Helper function for array indexing (equivalent to MATLAB's subarray)
def subarray(arr: np.ndarray, indices: Any) -> np.ndarray:
    """
    Extract subarray based on indices structure.
    This is a simplified version - full implementation would depend on
    the specific indexing structure used in MNPBEM.
    """
    return np.asarray(arr)