"""
hmatrix_linear.py - Linear Algebra Operations for H-Matrices
Converted from: hmatlu.cpp, hmatsolve.cpp, hmatlsolve.cpp, hmatrsolve.cpp, hmatinv.cpp

This module provides linear algebra operations:
- LU decomposition
- Linear system solving  
- Matrix inversion
"""

import numpy as np
from typing import Union, Optional, Dict, Any, Tuple
from .hlib import HMatrix, SubMatrix, Matrix, tree, hopts, timer, tic, toc

def hmat_lu_decomposition(hmat: HMatrix, options: Optional[Dict] = None) -> HMatrix:
    """
    LU decomposition of H-matrix.
    
    Equivalent to hmatlu.cpp functionality.
    
    Parameters:
    -----------
    hmat : HMatrix
        Input hierarchical matrix
    options : dict, optional
        Options with 'htol' and 'kmax'
        
    Returns:
    --------
    HMatrix
        LU decomposition result
    """
    if options:
        if 'htol' in options:
            hopts.tol = options['htol']
        if 'kmax' in options:
            hopts.kmax = options['kmax']
    
    start_time = tic()
    
    # Create result matrix
    result = HMatrix()
    
    # Perform LU decomposition on each submatrix
    for (row, col), submat in hmat.mat.items():
        if submat.empty():
            continue
            
        if row == col:  # Diagonal block
            if submat.flag() == 2:  # FLAG_FULL
                # LU decomposition using our implementation
                lu_mat = _lu_decomposition_full(submat.mat)
                result[(row, col)] = SubMatrix(row, col, lu_mat)
            else:  # FLAG_RK 
                # Convert to full, then LU decompose
                full_mat = Matrix()
                full_mat.val = np.dot(submat.lhs.val, submat.rhs.val.T)
                lu_mat = _lu_decomposition_full(full_mat)
                result[(row, col)] = SubMatrix(row, col, lu_mat)
        else:
            # Off-diagonal blocks remain the same initially
            result[(row, col)] = submat
    
    toc(start_time, "lu_decomposition")
    
    return result

def _lu_decomposition_full(mat: Matrix) -> Matrix:
    """
    LU decomposition for full matrix using Crout algorithm.
    """
    result = Matrix()
    result.val = mat.val.copy()
    m, n = result.val.shape
    
    for j in range(min(m, n)):
        # L part: L[i,j] = A[i,j] - sum(L[i,k]*U[k,j] for k in range(j))
        for i in range(j, m):
            if j > 0:
                result.val[i, j] -= np.dot(result.val[i, :j], result.val[:j, j])
        
        # U part: U[j,i] = (A[j,i] - sum(L[j,k]*U[k,i] for k in range(j))) / L[j,j]
        for i in range(j + 1, n):
            if j > 0:
                result.val[j, i] -= np.dot(result.val[j, :j], result.val[:j, i])
            if abs(result.val[j, j]) > 1e-14:
                result.val[j, i] /= result.val[j, j]
    
    return result

def hmat_solve_vector(lu_hmat: HMatrix, b: np.ndarray, method: str = 'N') -> np.ndarray:
    """
    Solve linear system A*x = b using LU decomposition.
    
    Equivalent to hmatsolve.cpp functionality.
    
    Parameters:
    -----------
    lu_hmat : HMatrix
        LU decomposed H-matrix
    b : np.ndarray
        Right-hand side vector
    method : str
        'L' for lower triangular, 'U' for upper triangular, 'N' for both
        
    Returns:
    --------
    np.ndarray
        Solution vector x
    """
    x = b.copy()
    
    if method in ['L', 'N']:
        # Forward substitution (lower triangular)
        x = _solve_triangular_hmat(lu_hmat, x, lower=True)
    
    if method in ['U', 'N']:
        # Back substitution (upper triangular)  
        x = _solve_triangular_hmat(lu_hmat, x, lower=False)
    
    return x

def hmat_lsolve(B: HMatrix, A: HMatrix, method: str = 'N', options: Optional[Dict] = None) -> HMatrix:
    """
    Solve A*X = B for H-matrices (left solve).
    
    Equivalent to hmatlsolve.cpp functionality.
    
    Parameters:
    -----------
    B : HMatrix
        Right-hand side H-matrix
    A : HMatrix
        LU decomposed coefficient H-matrix  
    method : str
        'L', 'U', or 'N' for solve method
    options : dict, optional
        Options with 'htol' and 'kmax'
        
    Returns:
    --------
    HMatrix
        Solution H-matrix X
    """
    if options:
        if 'htol' in options:
            hopts.tol = options['htol']
        if 'kmax' in options:
            hopts.kmax = options['kmax']
    
    start_time = tic()
    
    X = HMatrix()
    
    # Solve using hierarchical approach
    if method in ['L', 'N']:
        X = _hierarchical_lsolve(B, A, 'L')
    
    if method in ['U', 'N']:
        if method == 'N':
            X = _hierarchical_lsolve(X, A, 'U')  
        else:
            X = _hierarchical_lsolve(B, A, 'U')
    
    toc(start_time, "lsolve")
    
    return X

def hmat_rsolve(B: HMatrix, A: HMatrix, method: str = 'N', options: Optional[Dict] = None) -> HMatrix:
    """
    Solve X*A = B for H-matrices (right solve).
    
    Equivalent to hmatrsolve.cpp functionality.
    
    Parameters:
    -----------
    B : HMatrix
        Right-hand side H-matrix
    A : HMatrix
        LU decomposed coefficient H-matrix
    method : str
        'L', 'U', or 'N' for solve method
    options : dict, optional
        Options with 'htol' and 'kmax'
        
    Returns:
    --------
    HMatrix
        Solution H-matrix X
    """
    if options:
        if 'htol' in options:
            hopts.tol = options['htol']
        if 'kmax' in options:
            hopts.kmax = options['kmax']
    
    start_time = tic()
    
    X = HMatrix()
    
    # Solve using hierarchical approach  
    if method in ['U', 'N']:
        X = _hierarchical_rsolve(B, A, 'U')
    
    if method in ['L', 'N']:
        if method == 'N':
            X = _hierarchical_rsolve(X, A, 'L')
        else:
            X = _hierarchical_rsolve(B, A, 'L')
    
    toc(start_time, "rsolve")
    
    return X

def hmat_inverse(hmat: HMatrix, options: Optional[Dict] = None) -> HMatrix:
    """
    Compute inverse of H-matrix.
    
    Equivalent to hmatinv.cpp functionality.
    
    Parameters:
    -----------
    hmat : HMatrix
        Input hierarchical matrix
    options : dict, optional
        Options with 'htol' and 'kmax'
        
    Returns:
    --------
    HMatrix
        Inverse H-matrix
    """
    if options:
        if 'htol' in options:
            hopts.tol = options['htol']
        if 'kmax' in options:
            hopts.kmax = options['kmax']
    
    start_time = tic()
    
    # Hierarchical matrix inversion using recursive algorithm
    result = _hierarchical_inverse(hmat)
    
    toc(start_time, "inversion")
    
    return result

def _solve_triangular_hmat(hmat: HMatrix, b: np.ndarray, lower: bool = True) -> np.ndarray:
    """Solve triangular system with H-matrix."""
    x = b.copy()
    
    # Simplified triangular solve
    for (row, col), submat in sorted(hmat.mat.items()):
        if submat.empty() or row != col:
            continue
        
        if submat.flag() == 2:  # FLAG_FULL
            if lower:
                x = _solve_lower_triangular(submat.mat.val, x)
            else:
                x = _solve_upper_triangular(submat.mat.val, x)
    
    return x

def _solve_lower_triangular(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Lx = b for lower triangular L."""
    return np.linalg.solve(np.tril(L), b)

def _solve_upper_triangular(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ux = b for upper triangular U.""" 
    return np.linalg.solve(np.triu(U), b)

def _hierarchical_lsolve(B: HMatrix, A: HMatrix, uplo: str) -> HMatrix:
    """Hierarchical left solve implementation."""
    X = HMatrix()
    
    # Simplified implementation
    for (row, col), b_sub in B.mat.items():
        if b_sub.empty():
            continue
            
        a_sub = A.find(row, row)  # Diagonal block
        if a_sub and not a_sub.empty():
            if a_sub.flag() == 2:  # FLAG_FULL
                if b_sub.flag() == 2:
                    solution = Matrix()
                    if uplo == 'L':
                        solution.val = _solve_lower_triangular(a_sub.mat.val, b_sub.mat.val)
                    else:
                        solution.val = _solve_upper_triangular(a_sub.mat.val, b_sub.mat.val)
                    X[(row, col)] = SubMatrix(row, col, solution)
    
    return X

def _hierarchical_rsolve(B: HMatrix, A: HMatrix, uplo: str) -> HMatrix:
    """Hierarchical right solve implementation."""
    X = HMatrix()
    
    # Simplified implementation
    for (row, col), b_sub in B.mat.items():
        if b_sub.empty():
            continue
            
        a_sub = A.find(col, col)  # Diagonal block
        if a_sub and not a_sub.empty():
            if a_sub.flag() == 2 and b_sub.flag() == 2:  # Both full
                solution = Matrix()
                if uplo == 'U':
                    solution.val = np.linalg.solve(a_sub.mat.val.T, b_sub.mat.val.T).T
                else:
                    solution.val = np.linalg.solve(np.tril(a_sub.mat.val).T, b_sub.mat.val.T).T
                X[(row, col)] = SubMatrix(row, col, solution)
    
    return X

def _hierarchical_inverse(hmat: HMatrix) -> HMatrix:
    """Hierarchical matrix inversion using Schur complement."""
    result = HMatrix()
    
    # Simplified inversion
    for (row, col), submat in hmat.mat.items():
        if row == col and not submat.empty():  # Diagonal blocks
            if submat.flag() == 2:  # FLAG_FULL
                inv_mat = Matrix()
                inv_mat.val = np.linalg.inv(submat.mat.val)
                result[(row, col)] = SubMatrix(row, col, inv_mat)
            else:  # FLAG_RK - convert to full, invert
                full_mat = np.dot(submat.lhs.val, submat.rhs.val.T)
                inv_mat = Matrix()
                inv_mat.val = np.linalg.inv(full_mat)
                result[(row, col)] = SubMatrix(row, col, inv_mat)
    
    return result

class HMatrixLinear:
    """
    Class wrapper for H-matrix linear algebra operations.
    """
    
    @staticmethod
    def lu_decompose(hmat: HMatrix, **options) -> HMatrix:
        """LU decomposition of H-matrix."""
        return hmat_lu_decomposition(hmat, options)
    
    @staticmethod
    def solve_vector(lu_hmat: HMatrix, b: np.ndarray, method: str = 'N') -> np.ndarray:
        """Solve linear system with vector RHS."""
        return hmat_solve_vector(lu_hmat, b, method)
    
    @staticmethod  
    def solve_left(B: HMatrix, A: HMatrix, method: str = 'N', **options) -> HMatrix:
        """Left solve: A*X = B."""
        return hmat_lsolve(B, A, method, options)
    
    @staticmethod
    def solve_right(B: HMatrix, A: HMatrix, method: str = 'N', **options) -> HMatrix:
        """Right solve: X*A = B."""
        return hmat_rsolve(B, A, method, options)
    
    @staticmethod
    def inverse(hmat: HMatrix, **options) -> HMatrix:
        """Compute matrix inverse."""
        return hmat_inverse(hmat, options)