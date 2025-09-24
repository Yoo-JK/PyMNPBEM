"""
callback_aca.py - Function Handle ACA Interface
Converted from: hmatfun.cpp

This module provides ACA interface for user-defined callback functions,
equivalent to MATLAB's function handle mechanism.
"""

import numpy as np
from typing import Callable, Optional, Dict, Tuple
from .hlib import ACAFunc, HMatrix, SubMatrix, Matrix, tree, hopts, aca

class CallbackACA(ACAFunc):
    """
    ACA functor for user-defined callback functions.
    
    Equivalent to the acamex template class from hmatfun.cpp.
    This allows users to provide custom matrix element computation functions.
    """
    
    def __init__(self, callback_function: Callable, is_complex: bool = False):
        """
        Initialize callback ACA functor.
        
        Parameters:
        -----------
        callback_function : callable
            Function that computes matrix elements.
            Should have signature: f(row_indices, col_indices) -> values
        is_complex : bool
            Whether the function returns complex values
        """
        self.callback_function = callback_function
        self.is_complex = is_complex
        self.row = 0
        self.col = 0
        self.siz = None
        
        # Data type
        self.dtype = np.complex128 if is_complex else np.float64
    
    def init(self, row_cluster: int, col_cluster: int):
        """Initialize cluster."""
        self.row = row_cluster
        self.col = col_cluster
        
        if tree and hasattr(tree, 'size'):
            # Get cluster sizes from tree
            row_size = tree.size(row_cluster)
            col_size = tree.size(col_cluster)
            # Simplified mask creation
            self.siz = type('Mask', (), {
                'rbegin': 0, 'rend': 10,
                'cbegin': 0, 'cend': 10,
                'nrows': lambda: 10,
                'ncols': lambda: 10
            })()
        else:
            # Default size
            self.siz = type('Mask', (), {
                'rbegin': 0, 'rend': 10, 
                'cbegin': 0, 'cend': 10,
                'nrows': lambda: 10, 
                'ncols': lambda: 10
            })()
    
    def nrows(self) -> int:
        """Number of rows in current cluster block."""
        return self.siz.nrows() if self.siz else 1
    
    def ncols(self) -> int:
        """Number of columns in current cluster block."""
        return self.siz.ncols() if self.siz else 1
    
    def max_rank(self) -> int:
        """Maximum rank for low-rank approximation."""
        return min(self.nrows(), self.ncols())
    
    def getrow(self, r: int) -> np.ndarray:
        """
        Get row r of the matrix using callback function.
        
        Parameters:
        -----------
        r : int
            Row index (local to current cluster)
            
        Returns:
        --------
        np.ndarray
            Row vector of matrix values
        """
        n = self.ncols()
        
        # Create row and column index arrays (convert to 1-based for MATLAB compatibility)
        row_indices = np.full(n, self.siz.rbegin + r + 1, dtype=np.uint64)
        col_indices = np.arange(self.siz.cbegin + 1, self.siz.cbegin + n + 1, dtype=np.uint64)
        
        # Call user function
        try:
            values = self.callback_function(row_indices, col_indices)
            values = np.asarray(values, dtype=self.dtype)
            
            if values.size != n:
                raise ValueError(f"Callback function returned {values.size} values, expected {n}")
                
            return values.flatten()
            
        except Exception as e:
            raise RuntimeError(f"Error in callback function (getrow): {e}")
    
    def getcol(self, c: int) -> np.ndarray:
        """
        Get column c of the matrix using callback function.
        
        Parameters:
        -----------
        c : int
            Column index (local to current cluster)
            
        Returns:
        --------
        np.ndarray
            Column vector of matrix values
        """
        m = self.nrows()
        
        # Create row and column index arrays (convert to 1-based for MATLAB compatibility)  
        row_indices = np.arange(self.siz.rbegin + 1, self.siz.rbegin + m + 1, dtype=np.uint64)
        col_indices = np.full(m, self.siz.cbegin + c + 1, dtype=np.uint64)
        
        # Call user function
        try:
            values = self.callback_function(row_indices, col_indices)
            values = np.asarray(values, dtype=self.dtype)
            
            if values.size != m:
                raise ValueError(f"Callback function returned {values.size} values, expected {m}")
                
            return values.flatten()
            
        except Exception as e:
            raise RuntimeError(f"Error in callback function (getcol): {e}")

def hmat_callback_aca(callback_function: Callable, is_complex: bool, i: int, j: int, 
                     options: Optional[Dict] = None) -> Tuple[list, list]:
    """
    Fill H-matrix using user-defined callback function and ACA.
    
    Equivalent to hmatfun.cpp mexFunction.
    
    Parameters:
    -----------
    callback_function : callable
        User function that computes matrix elements
        Signature: f(row_indices, col_indices) -> values
    is_complex : bool
        Whether function returns complex values
    i, j : int
        Starting cluster indices
    options : dict, optional
        Options with 'htol' and 'kmax'
        
    Returns:
    --------
    tuple
        (L_cells, R_cells) - Low-rank factor cell arrays
    """
    if options:
        if 'htol' in options:
            hopts.tol = options['htol']
        if 'kmax' in options:
            hopts.kmax = options['kmax']
    
    # Set up ACA object with callback
    aca_obj = CallbackACA(callback_function, is_complex)
    aca_obj.init(i, j)
    
    # Compute low-rank approximation using ACA
    L, R = aca(aca_obj, hopts.tol)
    
    # Return as cell arrays
    L_cells = [L.val] if hasattr(L, 'val') else [L]
    R_cells = [R.val] if hasattr(R, 'val') else [R]
    
    return L_cells, R_cells

class FunctionHandleACA:
    """
    Class wrapper for function handle ACA operations.
    """
    
    @staticmethod
    def from_callback(callback_function: Callable, is_complex: bool = False) -> CallbackACA:
        """Create CallbackACA object from user function."""
        return CallbackACA(callback_function, is_complex)
    
    @staticmethod
    def compute_aca(callback_function: Callable, i: int, j: int, 
                   is_complex: bool = False, **options) -> Tuple[list, list]:
        """Compute ACA using callback function."""
        return hmat_callback_aca(callback_function, is_complex, i, j, options)
    
    @staticmethod
    def create_test_function(matrix_data: np.ndarray) -> Callable:
        """
        Create a test callback function from matrix data.
        
        Parameters:
        -----------
        matrix_data : np.ndarray
            Full matrix data
            
        Returns:
        --------
        callable
            Callback function that extracts elements from the matrix
        """
        def test_callback(row_indices, col_indices):
            # Convert from 1-based (MATLAB) to 0-based (Python) indexing
            rows = np.asarray(row_indices, dtype=int) - 1
            cols = np.asarray(col_indices, dtype=int) - 1
            
            # Clip indices to matrix bounds
            rows = np.clip(rows, 0, matrix_data.shape[0] - 1)
            cols = np.clip(cols, 0, matrix_data.shape[1] - 1)
            
            # Extract values
            if len(rows) == 1:
                # Single row, multiple columns
                return matrix_data[rows[0], cols]
            elif len(cols) == 1:
                # Multiple rows, single column
                return matrix_data[rows, cols[0]]
            else:
                # Element-wise extraction
                return matrix_data[rows, cols]
        
        return test_callback
    
    @staticmethod
    def create_analytic_function(func_type: str = 'exp') -> Callable:
        """
        Create analytical test functions.
        
        Parameters:
        -----------
        func_type : str
            Type of function ('exp', 'rational', 'oscillatory')
            
        Returns:
        --------
        callable
            Analytical callback function
        """
        if func_type == 'exp':
            def exp_callback(row_indices, col_indices):
                rows = np.asarray(row_indices, dtype=float)
                cols = np.asarray(col_indices, dtype=float)
                
                if len(rows) == len(cols):
                    distances = np.abs(rows - cols)
                else:
                    distances = np.abs(rows[:, np.newaxis] - cols).flatten()
                
                return np.exp(-distances / 10.0)
            
            return exp_callback
            
        elif func_type == 'rational':
            def rational_callback(row_indices, col_indices):
                rows = np.asarray(row_indices, dtype=float)
                cols = np.asarray(col_indices, dtype=float)
                
                if len(rows) == len(cols):
                    distances = np.abs(rows - cols)
                else:
                    distances = np.abs(rows[:, np.newaxis] - cols).flatten()
                
                return 1.0 / (1.0 + distances)
            
            return rational_callback
        
        else:
            raise ValueError(f"Unknown function type: {func_type}")

# Convenience functions
def create_callback_aca(callback_function: Callable, is_complex: bool = False) -> CallbackACA:
    """Create callback ACA object."""
    return FunctionHandleACA.from_callback(callback_function, is_complex)

def compute_callback_aca(callback_function: Callable, i: int, j: int, 
                        is_complex: bool = False, **options) -> Tuple[list, list]:
    """Compute ACA using callback function."""
    return FunctionHandleACA.compute_aca(callback_function, i, j, is_complex, **options)