import numpy as np
from typing import Tuple, Callable, Union, Any

class IGrid2:
    """
    2D grid for interpolation using bilinear interpolation.
    
    This class provides functionality for bilinear interpolation on a 2D grid,
    including computation of derivatives along specified directions.
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Initialize 2D grid for interpolation.
        
        Parameters:
        -----------
        x : array_like
            x-values of grid points
        y : array_like  
            y-values of grid points
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        
        # Ensure arrays are sorted for proper interpolation
        if not np.all(np.diff(self.x) > 0):
            raise ValueError("x values must be strictly increasing")
        if not np.all(np.diff(self.y) > 0):
            raise ValueError("y values must be strictly increasing")
    
    @property
    def size(self) -> Tuple[int, int]:
        """Size of grid used for interpolation."""
        return (len(self.x), len(self.y))
    
    @property
    def numel(self) -> int:
        """Number of grid elements."""
        return len(self.x) * len(self.y)
    
    def finterp(self, x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create interpolation function for points (x,y).
        
        Parameters:
        -----------
        x, y : array_like
            Interpolation points
            
        Returns:
        --------
        callable
            Interpolation function that takes array V with tabulated values
            and returns interpolated values at positions (x,y)
        """
        x = np.asarray(x)
        y = np.asarray(y)
        x_flat = x.flatten()
        y_flat = y.flatten()
        
        # Find indices for grid positions using searchsorted (equivalent to histc)
        ix = np.searchsorted(self.x, x_flat, side='right') - 1
        iy = np.searchsorted(self.y, y_flat, side='right') - 1
        
        # Check bounds
        if np.any(ix < 0) or np.any(ix >= len(self.x) - 1):
            raise ValueError("x values are outside interpolation range")
        if np.any(iy < 0) or np.any(iy >= len(self.y) - 1):
            raise ValueError("y values are outside interpolation range")
        
        # Handle edge cases where x == max(self.x) or y == max(self.y)
        ix = np.minimum(ix, len(self.x) - 2)
        iy = np.minimum(iy, len(self.y) - 2)
        
        # Compute normalized coordinates within each grid cell
        xx = (x_flat - self.x[ix]) / (self.x[ix + 1] - self.x[ix])
        yy = (y_flat - self.y[iy]) / (self.y[iy + 1] - self.y[iy])
        
        # Grid size for computing linear indices
        siz = (len(self.x), len(self.y))
        
        # Compute linear indices for the four corners of each grid cell
        ind = np.column_stack([
            np.ravel_multi_index((ix, iy), siz),          # (i, j)
            np.ravel_multi_index((ix + 1, iy), siz),      # (i+1, j)
            np.ravel_multi_index((ix, iy + 1), siz),      # (i, j+1)
            np.ravel_multi_index((ix + 1, iy + 1), siz)   # (i+1, j+1)
        ])
        
        # Bilinear interpolation weights
        w = np.column_stack([
            (1 - xx) * (1 - yy),  # weight for (i, j)
            xx * (1 - yy),        # weight for (i+1, j)
            (1 - xx) * yy,        # weight for (i, j+1)
            xx * yy               # weight for (i+1, j+1)
        ])
        
        def interpolate(v: np.ndarray) -> np.ndarray:
            """Perform bilinear interpolation."""
            v_flat = v.flatten()
            result = np.sum(w * v_flat[ind], axis=1)
            return result.reshape(x.shape)
        
        return interpolate
    
    def fderiv(self, x: np.ndarray, y: np.ndarray, direction: int) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create derivative function for points (x,y) along specified direction.
        
        Parameters:
        -----------
        x, y : array_like
            Points where derivative is computed
        direction : int
            Derivative direction (1 for x, 2 for y)
            
        Returns:
        --------
        callable
            Function that computes derivative of interpolated values
        """
        x = np.asarray(x)
        y = np.asarray(y)
        x_flat = x.flatten()
        y_flat = y.flatten()
        
        # Find indices for grid positions
        ix = np.searchsorted(self.x, x_flat, side='right') - 1
        iy = np.searchsorted(self.y, y_flat, side='right') - 1
        
        # Check bounds
        if np.any(ix < 0) or np.any(ix >= len(self.x) - 1):
            raise ValueError("x values are outside interpolation range")
        if np.any(iy < 0) or np.any(iy >= len(self.y) - 1):
            raise ValueError("y values are outside interpolation range")
        
        # Handle edge cases
        ix = np.minimum(ix, len(self.x) - 2)
        iy = np.minimum(iy, len(self.y) - 2)
        
        # Bin sizes and normalized coordinates
        hx = self.x[ix + 1] - self.x[ix]
        hy = self.y[iy + 1] - self.y[iy]
        xx = (x_flat - self.x[ix]) / hx
        yy = (y_flat - self.y[iy]) / hy
        
        # Grid size for computing linear indices
        siz = (len(self.x), len(self.y))
        
        # Compute linear indices for the four corners
        ind = np.column_stack([
            np.ravel_multi_index((ix, iy), siz),
            np.ravel_multi_index((ix + 1, iy), siz),
            np.ravel_multi_index((ix, iy + 1), siz),
            np.ravel_multi_index((ix + 1, iy + 1), siz)
        ])
        
        # Derivative weights
        if direction == 1:  # derivative w.r.t. x
            w = np.column_stack([
                -(1 - yy) / hx,  # d/dx weight for (i, j)
                (1 - yy) / hx,   # d/dx weight for (i+1, j)
                -yy / hx,        # d/dx weight for (i, j+1)
                yy / hx          # d/dx weight for (i+1, j+1)
            ])
        elif direction == 2:  # derivative w.r.t. y
            w = np.column_stack([
                -(1 - xx) / hy,  # d/dy weight for (i, j)
                -xx / hy,        # d/dy weight for (i+1, j)
                (1 - xx) / hy,   # d/dy weight for (i, j+1)
                xx / hy          # d/dy weight for (i+1, j+1)
            ])
        else:
            raise ValueError("direction must be 1 (x) or 2 (y)")
        
        def derivative(v: np.ndarray) -> np.ndarray:
            """Compute derivative of interpolated values."""
            v_flat = v.flatten()
            result = np.sum(w * v_flat[ind], axis=1)
            return result.reshape(x.shape)
        
        return derivative
    
    def __call__(self, x: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Perform interpolation directly.
        
        Parameters:
        -----------
        x, y : array_like
            Interpolation points
        v : array_like
            Values at grid points (must have shape compatible with self.size)
            
        Returns:
        --------
        np.ndarray
            Interpolated values at positions (x, y)
        """
        interpolation_func = self.finterp(x, y)
        return interpolation_func(v)
    
    def __str__(self) -> str:
        """String representation."""
        return f"IGrid2(x: {len(self.x)} points, y: {len(self.y)} points)"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"IGrid2(\n"
                f"  x: [{self.x[0]:.3f}, ..., {self.x[-1]:.3f}] ({len(self.x)} points)\n"
                f"  y: [{self.y[0]:.3f}, ..., {self.y[-1]:.3f}] ({len(self.y)} points)\n"
                f")")


# Helper function for array indexing (equivalent to MATLAB's subarray)
def subarray(arr: np.ndarray, indices: Any) -> np.ndarray:
    """
    Extract subarray based on indices structure.
    This is a simplified version - full implementation would depend on
    the specific indexing structure used in MNPBEM.
    """
    return np.asarray(arr)