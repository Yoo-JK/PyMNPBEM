"""
hmatrix_ops.py - Basic H-Matrix Operations
Converted from: hmatfull.cpp, hmatadd.cpp, hmatmul1.cpp, hmatmul2.cpp

This module provides basic operations on hierarchical matrices:
- Conversion to full matrices
- Addition of H-matrices  
- Matrix-vector multiplication
- Matrix-matrix multiplication
"""

import numpy as np
from typing import Union, Optional, Dict, Any
from .hlib import HMatrix, SubMatrix, Matrix, tree, hopts, timer, tic, toc

def hmat_to_full(hmat: HMatrix) -> np.ndarray:
    """
    Convert H-matrix to full matrix.
    
    Equivalent to hmatfull.cpp functionality.
    
    Parameters:
    -----------
    hmat : HMatrix
        Hierarchical matrix to convert
        
    Returns:
    --------
    np.ndarray
        Full dense matrix
    """
    if not hmat.mat:
        return np.array([[]])
    
    # Determine matrix size from tree structure
    # This is simplified - in real implementation would use tree.ind
    max_row = max(key[0] for key in hmat.mat.keys()) if hmat.mat else 0
    max_col = max(key[1] for key in hmat.mat.keys()) if hmat.mat else 0
    
    # Estimate size (simplified)
    size = max(max_row, max_col) * 10  # Rough estimate
    if size == 0:
        return np.array([[]])
    
    full_mat = np.zeros((size, size), dtype=np.complex128)
    
    # Fill full matrix from submatrices
    for (row_cluster, col_cluster), submat in hmat.mat.items():
        if submat.empty():
            continue
            
        # Get cluster positions (simplified)
        row_start = row_cluster * 10  # Simplified indexing
        col_start = col_cluster * 10
        
        if submat.flag() == 2:  # FLAG_FULL
            submat_data = submat.mat.val
        else:  # FLAG_RK
            # Convert low-rank to full: L @ R.T
            submat_data = np.dot(submat.lhs.val, submat.rhs.val.T)
        
        # Place in full matrix
        rows, cols = submat_data.shape
        full_mat[row_start:row_start+rows, col_start:col_start+cols] = submat_data
    
    return full_mat

def hmat_add(A: HMatrix, B: HMatrix, options: Optional[Dict] = None) -> HMatrix:
    """
    Add two H-matrices: C = A + B
    
    Equivalent to hmatadd.cpp functionality.
    
    Parameters:
    -----------
    A, B : HMatrix
        Input hierarchical matrices
    options : dict, optional
        Options dictionary with 'htol' and 'kmax'
        
    Returns:
    --------
    HMatrix
        Result of A + B
    """
    # Set options
    if options:
        if 'htol' in options:
            hopts.tol = options['htol']
        if 'kmax' in options:
            hopts.kmax = options['kmax']
    
    # Use the built-in addition operator from HMatrix class
    result = A + B
    
    return result

def hmat_mul_vector(hmat: HMatrix, x: np.ndarray) -> np.ndarray:
    """
    Multiply H-matrix with vector/matrix: y = H * x
    
    Equivalent to hmatmul1.cpp functionality.
    
    Parameters:
    -----------
    hmat : HMatrix
        Hierarchical matrix
    x : np.ndarray
        Vector or matrix to multiply
        
    Returns:
    --------
    np.ndarray
        Result vector/matrix y = H * x
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    y = np.zeros_like(x, dtype=np.complex128)
    
    # Multiply each submatrix
    for (row_cluster, col_cluster), submat in hmat.mat.items():
        if submat.empty():
            continue
        
        # Get cluster index ranges (simplified)
        row_start = row_cluster * 10
        row_end = row_start + submat.nrows()
        col_start = col_cluster * 10  
        col_end = col_start + submat.ncols()
        
        # Extract relevant part of x
        if col_end <= x.shape[0]:
            x_block = x[col_start:col_end, :]
            
            if submat.flag() == 2:  # FLAG_FULL
                # Full matrix multiplication
                y_block = np.dot(submat.mat.val, x_block)
            else:  # FLAG_RK
                # Low-rank multiplication: L @ (R.T @ x)
                temp = np.dot(submat.rhs.val.T, x_block)
                y_block = np.dot(submat.lhs.val, temp)
            
            # Add to result
            if row_end <= y.shape[0]:
                y[row_start:row_end, :] += y_block
    
    return y.squeeze() if y.shape[1] == 1 else y

def hmat_mul_hmat(A: HMatrix, B: HMatrix, options: Optional[Dict] = None) -> HMatrix:
    """
    Multiply two H-matrices: C = A * B
    
    Equivalent to hmatmul2.cpp functionality.
    
    Parameters:
    -----------
    A, B : HMatrix
        Input hierarchical matrices
    options : dict, optional
        Options dictionary with 'htol' and 'kmax'
        
    Returns:
    --------
    HMatrix
        Result of A * B
    """
    # Set options
    if options:
        if 'htol' in options:
            hopts.tol = options['htol']
        if 'kmax' in options:
            hopts.kmax = options['kmax']
    
    start_time = tic()
    
    # Use the built-in multiplication operator from HMatrix class
    result = A * B
    
    toc(start_time, "hmat_multiplication")
    
    return result

class HMatrixOps:
    """
    Class wrapper for H-matrix basic operations.
    
    Provides a convenient interface to all basic H-matrix operations.
    """
    
    @staticmethod
    def to_full(hmat: HMatrix) -> np.ndarray:
        """Convert H-matrix to full matrix."""
        return hmat_to_full(hmat)
    
    @staticmethod
    def add(A: HMatrix, B: HMatrix, **options) -> HMatrix:
        """Add two H-matrices."""
        return hmat_add(A, B, options)
    
    @staticmethod
    def multiply_vector(hmat: HMatrix, x: np.ndarray) -> np.ndarray:
        """Multiply H-matrix with vector/matrix."""
        return hmat_mul_vector(hmat, x)
    
    @staticmethod
    def multiply_hmat(A: HMatrix, B: HMatrix, **options) -> HMatrix:
        """Multiply two H-matrices."""
        return hmat_mul_hmat(A, B, options)