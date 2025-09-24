"""
hlib - Hierarchical Matrix Library for MNPBEM
Converted from MATLAB MEX hlib folder

This module implements:
- Base matrix classes and operations (basemat.cpp/h)
- Adaptive Cross Approximation (ACA) algorithm (aca.cpp/h)
- Hierarchical matrices (hmatrix.h)
- Cluster trees (clustertree.h)
- LU decomposition for H-matrices (lu.cpp/h)
- Submatrix operations (submatrix.h)
"""

import numpy as np
from typing import Tuple, Optional, Union, Dict, Any, List
from abc import ABC, abstractmethod
import warnings
from scipy.linalg import norm, solve_triangular
import time

# Global options
class HOptions:
    """Options for hierarchical matrices."""
    def __init__(self, tol: float = 1e-6, kmax: int = 50):
        self.tol = tol      # Tolerance for truncation of Rk matrices
        self.kmax = kmax    # Maximum rank for low-rank matrices

hopts = HOptions()

# Global timer
timer = {}

def tic():
    """Start timer."""
    return time.time()

def toc(start_time, name: str):
    """End timer and add to global timer dict."""
    elapsed = time.time() - start_time
    if name in timer:
        timer[name] += elapsed
    else:
        timer[name] = elapsed

# Constants for matrix types
FLAG_RK = 1
FLAG_FULL = 2
ACATOL = 1e-10

class Mask:
    """Mask for matrix indexing."""
    def __init__(self, rbegin: int, rend: int, cbegin: int, cend: int):
        self.rbegin = rbegin
        self.rend = rend
        self.cbegin = cbegin
        self.cend = cend
    
    def nrows(self):
        return self.rend - self.rbegin
    
    def ncols(self):
        return self.cend - self.cbegin

class Matrix:
    """Enhanced matrix class with BLAS-like operations."""
    
    def __init__(self, data=None, shape=None, fill_value=None):
        if data is not None:
            self.val = np.asarray(data, dtype=np.complex128)
        elif shape is not None:
            if fill_value is not None:
                self.val = np.full(shape, fill_value, dtype=np.complex128)
            else:
                self.val = np.zeros(shape, dtype=np.complex128)
        else:
            self.val = np.array([], dtype=np.complex128)
    
    def nrows(self):
        return self.val.shape[0] if self.val.ndim > 0 else 0
    
    def ncols(self):
        return self.val.shape[1] if self.val.ndim > 1 else 1
    
    def empty(self):
        return self.val.size == 0
    
    def size(self):
        return Mask(0, self.nrows(), 0, self.ncols())
    
    def copy(self, other):
        """Copy from another matrix."""
        self.val = np.copy(other.val)
        return self
    
    def add_to(self, other, alpha=1.0):
        """Add scaled matrix: self += alpha * other."""
        self.val += alpha * other.val
        return self
    
    def scale(self, alpha):
        """Scale matrix: self *= alpha.""" 
        self.val *= alpha
        return self
    
    def __add__(self, other):
        result = Matrix()
        result.val = self.val + other.val
        return result
    
    def __sub__(self, other):
        result = Matrix()
        result.val = self.val - other.val
        return result
    
    def __neg__(self):
        result = Matrix()
        result.val = -self.val
        return result
    
    def __mul__(self, other):
        """Matrix multiplication."""
        result = Matrix()
        result.val = np.dot(self.val, other.val)
        return result

def transpose(A: Matrix) -> Matrix:
    """Transpose matrix."""
    result = Matrix()
    result.val = A.val.T
    return result

def cat_matrices(*matrices):
    """Concatenate matrices horizontally."""
    vals = [m.val for m in matrices]
    result = Matrix()
    result.val = np.hstack(vals)
    return result

def matrix_inv(A: Matrix) -> Matrix:
    """Matrix inversion using NumPy."""
    result = Matrix()
    result.val = np.linalg.inv(A.val)
    return result

# ACA Base Classes
class ACAFunc(ABC):
    """Virtual ACA functor base class."""
    
    @abstractmethod
    def nrows(self) -> int:
        """Number of rows."""
        pass
    
    @abstractmethod
    def ncols(self) -> int:
        """Number of columns."""
        pass
    
    @abstractmethod
    def max_rank(self) -> int:
        """Maximum rank of low-rank matrices."""
        pass
    
    @abstractmethod
    def getrow(self, r: int) -> np.ndarray:
        """Get row r of the matrix."""
        pass
    
    @abstractmethod
    def getcol(self, c: int) -> np.ndarray:
        """Get column c of the matrix."""
        pass

class ACAFull(ACAFunc):
    """ACA functor for full matrix."""
    
    def __init__(self, mat: Matrix):
        self.mat = mat
    
    def nrows(self) -> int:
        return self.mat.nrows()
    
    def ncols(self) -> int:
        return self.mat.ncols()
    
    def max_rank(self) -> int:
        return min(self.nrows(), self.ncols())
    
    def getrow(self, r: int) -> np.ndarray:
        """Get row r from full matrix."""
        return self.mat.val[r, :].copy()
    
    def getcol(self, c: int) -> np.ndarray:
        """Get column c from full matrix."""
        return self.mat.val[:, c].copy()

class ACARk(ACAFunc):
    """ACA functor for low-rank matrix."""
    
    def __init__(self, lhs: Matrix, rhs: Matrix):
        self.lhs = lhs
        self.rhs = rhs
    
    def nrows(self) -> int:
        return self.lhs.nrows()
    
    def ncols(self) -> int:
        return self.rhs.nrows()
    
    def max_rank(self) -> int:
        return self.lhs.ncols()
    
    def getrow(self, r: int) -> np.ndarray:
        """Get row r from low-rank matrix: rhs.T @ lhs[r, :]."""
        return np.dot(self.rhs.val.T, self.lhs.val[r, :])
    
    def getcol(self, c: int) -> np.ndarray:
        """Get column c from low-rank matrix: lhs @ rhs[c, :]."""
        return np.dot(self.lhs.val, self.rhs.val[c, :])

def aca(fun: ACAFunc, tol: float = None) -> Tuple[Matrix, Matrix]:
    """
    Adaptive Cross Approximation algorithm.
    
    Parameters:
    -----------
    fun : ACAFunc
        Function object providing matrix access
    tol : float
        Tolerance for convergence
        
    Returns:
    --------
    L, R : Matrix
        Low-rank factors such that A ≈ L @ R.T
    """
    if tol is None:
        tol = hopts.tol
    
    m, n = fun.nrows(), fun.ncols()
    kmax = min(fun.max_rank(), hopts.kmax)
    
    # Build up low-rank approximation A ≈ A @ B.T using ACA
    A = Matrix(shape=(m, kmax), fill_value=0.0)
    B = Matrix(shape=(n, kmax), fill_value=0.0)
    
    # Summed up norm and new norm
    Nsum = 0.0
    
    # Vector for pivot elements
    row_indices = list(range(m))
    r = 0  # Current pivot row
    
    start_time = tic()
    
    # ACA loop
    for k in range(kmax):
        # Fill row B[:, k]
        row_k = fun.getrow(r)
        if np.linalg.norm(row_k) < ACATOL:
            break
            
        B.val[:, k] = row_k
        
        # Subtract current approximation A @ B.T
        if k > 0:
            approx = np.dot(A.val[r, :k], B.val[:, :k].T)
            B.val[:, k] -= approx
        
        # Find pivot column c
        c = np.argmax(np.abs(B.val[:, k]))
        
        # Scale B[:, k]
        if abs(B.val[c, k]) > 1e-14:
            scale = 1.0 / B.val[c, k]
            B.val[:, k] *= scale
        else:
            break
        
        # Fill column A[:, k]  
        col_k = fun.getcol(c)
        A.val[:, k] = col_k
        
        # Subtract current approximation A @ B.T
        if k > 0:
            approx = np.dot(A.val[:, :k], B.val[c, :k])
            A.val[:, k] -= approx
        
        # Remove current pivot row and find next pivot row
        if r in row_indices:
            row_indices.remove(r)
        
        if not row_indices:
            break
            
        # Find next pivot row with maximum absolute value
        r = max(row_indices, key=lambda i: abs(A.val[i, k]))
        
        # Norm of new vector elements
        Nk = np.linalg.norm(A.val[:, k]) * np.linalg.norm(B.val[:, k])
        
        # Check for convergence
        if Nk < tol * Nsum or not row_indices:
            break
        else:
            Nsum = np.sqrt(Nsum**2 + Nk**2)
    
    toc(start_time, "aca")
    
    # Set output (trim to actual rank)
    actual_rank = min(k + 1, kmax)
    L = Matrix()
    L.val = A.val[:, :actual_rank].copy()
    R = Matrix()
    R.val = B.val[:, :actual_rank].copy()
    
    return L, R

# Cluster Tree (simplified version)
class ClusterTree:
    """Cluster tree for hierarchical matrices."""
    
    def __init__(self):
        self.sons = {}      # Dictionary of sons for each cluster
        self.ind = {}       # Index ranges for each cluster
        self.ipart = {}     # Particle indices
        self.ad = {}        # Admissibility map
    
    def size(self, i: int) -> Tuple[int, int]:
        """Size of cluster i."""
        if i in self.ind:
            return self.ind[i]
        return (0, 1)  # Default
    
    def leaf(self, ic: int) -> bool:
        """Determine whether cluster is leaf."""
        return ic not in self.sons or (self.sons[ic][0] == 0 and self.sons[ic][1] == 0)
    
    def admiss(self, row: int, col: int) -> int:
        """Admissibility of cluster pairs."""
        pair = (row, col)
        return self.ad.get(pair, 0)
    
    def is_admissible(self, row: int, col: int) -> bool:
        """Check if cluster pair is admissible for low-rank approximation."""
        return self.admiss(row, col) == FLAG_RK

# Global tree instance
tree = ClusterTree()

# Submatrix class
class SubMatrix:
    """Sub-matrix (full or low-rank) for hierarchical matrices."""
    
    def __init__(self, row: int = 0, col: int = 0, mat: Matrix = None, lhs: Matrix = None, rhs: Matrix = None):
        self.row = row
        self.col = col
        
        if mat is not None:
            self.mat = mat
            self.lhs = Matrix()  # Empty
            self.rhs = Matrix()  # Empty
        elif lhs is not None and rhs is not None:
            self.mat = Matrix()  # Empty
            self.lhs = lhs
            self.rhs = rhs
        else:
            self.mat = Matrix()
            self.lhs = Matrix()
            self.rhs = Matrix()
    
    def nrows(self) -> int:
        """Number of rows."""
        if not self.mat.empty():
            return self.mat.nrows()
        elif not self.lhs.empty():
            return self.lhs.nrows()
        return 0
    
    def ncols(self) -> int:
        """Number of columns."""
        if not self.mat.empty():
            return self.mat.ncols()
        elif not self.rhs.empty():
            return self.rhs.nrows()
        return 0
    
    def empty(self) -> bool:
        """Check if submatrix is empty."""
        return self.mat.empty() and self.lhs.empty()
    
    def flag(self) -> int:
        """Get matrix type flag."""
        if self.empty():
            return 0
        elif self.lhs.empty():
            return FLAG_FULL
        else:
            return FLAG_RK
    
    def name(self) -> str:
        """Get matrix type name."""
        flag = self.flag()
        if flag == FLAG_FULL:
            return "full"
        elif flag == FLAG_RK:
            return "Rk"
        else:
            return ""
    
    def rank(self) -> int:
        """Rank of submatrix."""
        if self.flag() == FLAG_RK:
            return self.lhs.ncols()
        return min(self.nrows(), self.ncols())
    
    def convert(self, cflag: int):
        """Convert storage format."""
        if self.flag() == cflag:
            return self
        elif self.flag() == FLAG_RK and cflag == FLAG_FULL:
            # Convert low-rank to full
            full_mat = Matrix()
            full_mat.val = np.dot(self.lhs.val, self.rhs.val.T)
            return SubMatrix(self.row, self.col, full_mat)
        elif self.flag() == FLAG_FULL and cflag == FLAG_RK:
            # Convert full to low-rank using ACA
            aca_fun = ACAFull(self.mat)
            L, R = aca(aca_fun)
            return SubMatrix(self.row, self.col, lhs=L, rhs=R)
        
        return self
    
    def __add__(self, other):
        """Addition of submatrices."""
        if self.empty():
            return other
        elif other.empty():
            return self
        elif self.flag() == FLAG_FULL and other.flag() == FLAG_FULL:
            # Full + Full
            result_mat = self.mat + other.mat
            return SubMatrix(self.row, self.col, result_mat)
        else:
            # Convert to low-rank and concatenate
            self_rk = self.convert(FLAG_RK)
            other_rk = other.convert(FLAG_RK)
            
            # Concatenate factors
            new_lhs = Matrix()
            new_lhs.val = np.hstack([self_rk.lhs.val, other_rk.lhs.val])
            new_rhs = Matrix()
            new_rhs.val = np.hstack([self_rk.rhs.val, other_rk.rhs.val])
            
            result = SubMatrix(self.row, self.col, lhs=new_lhs, rhs=new_rhs)
            
            # Truncate using ACA
            return truncate_submatrix(result)
    
    def __neg__(self):
        """Unary minus."""
        if self.flag() == FLAG_FULL:
            return SubMatrix(self.row, self.col, -self.mat)
        else:
            return SubMatrix(self.row, self.col, lhs=-self.lhs, rhs=self.rhs)

def truncate_submatrix(A: SubMatrix) -> SubMatrix:
    """Truncate low-rank matrix using ACA."""
    if A.flag() == FLAG_RK:
        aca_fun = ACARk(A.lhs, A.rhs)
        L, R = aca(aca_fun)
        return SubMatrix(A.row, A.col, lhs=L, rhs=R)
    return A

# Hierarchical Matrix class
class HMatrix:
    """Hierarchical matrix implementation."""
    
    def __init__(self):
        self.mat = {}  # Dictionary mapping (row, col) pairs to SubMatrix
    
    def find(self, row: int, col: int) -> Optional[SubMatrix]:
        """Find submatrix at (row, col)."""
        return self.mat.get((row, col), None)
    
    def __setitem__(self, key: Tuple[int, int], value: SubMatrix):
        """Set submatrix."""
        self.mat[key] = value
    
    def __getitem__(self, key: Tuple[int, int]) -> SubMatrix:
        """Get submatrix."""
        return self.mat.get(key, SubMatrix())
    
    def clear(self):
        """Clear H-matrix."""
        self.mat.clear()
    
    def __add__(self, other):
        """H-matrix addition."""
        result = HMatrix()
        
        # Add all submatrices
        all_keys = set(self.mat.keys()) | set(other.mat.keys())
        for key in all_keys:
            A = self.mat.get(key, SubMatrix())
            B = other.mat.get(key, SubMatrix())
            if not A.empty() or not B.empty():
                result[key] = A + B
        
        return result
    
    def __neg__(self):
        """Unary minus."""
        result = HMatrix()
        for key, submat in self.mat.items():
            result[key] = -submat
        return result
    
    def __sub__(self, other):
        """H-matrix subtraction."""
        return self + (-other)
    
    def __mul__(self, other):
        """Matrix-vector or H-matrix multiplication."""
        if isinstance(other, np.ndarray):
            # H-matrix * vector
            y = np.zeros_like(other)
            for (row, col), submat in self.mat.items():
                # This is simplified - would need proper indexing with tree
                if submat.flag() == FLAG_FULL:
                    y += np.dot(submat.mat.val, other)
                elif submat.flag() == FLAG_RK:
                    temp = np.dot(submat.rhs.val.T, other)
                    y += np.dot(submat.lhs.val, temp)
            return y
        elif isinstance(other, HMatrix):
            # H-matrix * H-matrix (simplified)
            result = HMatrix()
            # This would require proper tree traversal
            return result
    
    def items(self):
        """Iterator over (key, submatrix) pairs."""
        return self.mat.items()

# Convenience functions
def aca_full_matrix(mat: Matrix, tol: float = None) -> Tuple[Matrix, Matrix]:
    """ACA for full matrix."""
    return aca(ACAFull(mat), tol)

def aca_low_rank(lhs: Matrix, rhs: Matrix, tol: float = None) -> Tuple[Matrix, Matrix]:
    """ACA for low-rank matrix (recompression)."""
    return aca(ACARk(lhs, rhs), tol)

# LU Decomposition functions (simplified)
def lu_decomposition(A: Matrix) -> Matrix:
    """LU decomposition using custom Crout algorithm."""
    B = Matrix()
    B.val = A.val.copy()
    m, n = B.nrows(), B.ncols()
    
    start_time = tic()
    
    for j in range(min(m, n)):
        # L part: L[i,j] = A[i,j] - sum(L[i,k]*U[k,j] for k in range(j))
        for i in range(j, m):
            if j > 0:
                B.val[i, j] -= np.dot(B.val[i, :j], B.val[:j, j])
        
        # U part: U[j,i] = (A[j,i] - sum(L[j,k]*U[k,i] for k in range(j))) / L[j,j]
        for i in range(j + 1, n):
            if j > 0:
                B.val[j, i] -= np.dot(B.val[j, :j], B.val[:j, i])
            if abs(B.val[j, j]) > 1e-14:
                B.val[j, i] /= B.val[j, j]
    
    toc(start_time, "lu")
    return B

def solve_triangular_system(A: Matrix, b: Matrix, lower: bool = True) -> Matrix:
    """Solve triangular system."""
    result = Matrix()
    result.val = solve_triangular(A.val, b.val, lower=lower)
    return result

# Testing and utility functions
def full_matrix_from_hmatrix(H: HMatrix) -> Matrix:
    """Convert H-matrix to full matrix (for testing)."""
    # This would require proper tree structure to determine matrix size
    # For now, return placeholder
    return Matrix(shape=(1, 1), fill_value=0.0)

def print_timer():
    """Print timing information."""
    print("Timing information:")
    for name, time_val in timer.items():
        print(f"  {name}: {time_val:.4f} seconds")

# Initialize global options
hopts = HOptions(tol=1e-6, kmax=50)