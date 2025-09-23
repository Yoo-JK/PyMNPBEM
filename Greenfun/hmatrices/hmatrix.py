import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.linalg import qr, svd
import matplotlib.pyplot as plt
from ..bemoptions import get_bemoptions


class HMatrix:
    """
    Hierarchical matrix.
    Using a cluster tree and an admissibility matrix for the application
    of low-rank approximations, this class stores and manipulates
    low-rank hierarchical matrices.
    
    See S. Boerm et al., Eng. Analysis with Bound. Elem. 27, 405 (2003).
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize hierarchical matrix.
        
        Parameters
        ----------
        tree : object
            Cluster tree
        *args, **kwargs : optional
            Options and property pairs
            
        Properties
        ----------
        fadmiss : function, optional
            Function for admissibility, e.g.
            lambda rad1, rad2, dist: 2 * min(rad1, rad2) < dist
        htol : float, optional
            Tolerance for low-rank approximation
        """
        self.tree = None
        self.htol = 1e-6
        self.kmax = 100
        
        self.row1 = None  # rows for full-rank matrices
        self.col1 = None  # columns for full-rank matrices
        self.row2 = None  # rows for low-rank matrices
        self.col2 = None  # columns for low-rank matrices
        
        self.val = None   # full matrix
        self.lhs = None   # low-rank matrix lhs
        self.rhs = None   # low-rank matrix rhs (lhs * rhs.T)
        
        self.op = None    # option structure
        self.stat = None  # statistics
        
        if args:
            self._init(*args, **kwargs)
    
    def __str__(self):
        """Command window display."""
        info = {
            'tree': self.tree,
            'htol': self.htol,
            'kmax': self.kmax,
            'val': self.val,
            'lhs': self.lhs,
            'rhs': self.rhs
        }
        return f"HMatrix:\n{info}"
    
    def _init(self, tree, *args, **kwargs):
        """Initialize hierarchical matrix."""
        op = get_bemoptions(*args, **kwargs)
        
        self.tree = tree
        
        # Set tolerance and maximum rank
        if hasattr(op, 'htol'):
            self.htol = op.htol
        if hasattr(op, 'kmax'):
            self.kmax = op.kmax
        
        # Build admissibility matrix
        if hasattr(op, 'fadmiss'):
            fadmiss = op.fadmiss
        else:
            fadmiss = lambda rad1, rad2, dist: 2.5 * min(rad1, rad2) < dist
        
        # Get admissibility matrix
        admiss = tree.admissibility(tree, fadmiss=fadmiss)
        
        # Find full and low-rank matrices
        row1, col1 = np.where(admiss.toarray() == 2)
        row2, col2 = np.where(admiss.toarray() == 1)
        
        self.row1 = row1
        self.col1 = col1
        self.row2 = row2
        self.col2 = col2
        
        # Initialize storage
        self.val = [None] * len(row1)
        self.lhs = [None] * len(row2)
        self.rhs = [None] * len(row2)
        
        self.op = op
    
    def aca(self, fun):
        """
        Fill matrix with user-defined function using ACA.
        
        Parameters
        ----------
        fun : function
            User-defined function fun(row, col) that returns matrix values
            
        Returns
        -------
        HMatrix
            H-matrix with full and low-rank matrices
        """
        tree = self.tree
        ind = tree.ind[:, 1]  # transformation to cluster indices
        
        # Modify input function
        fun2 = lambda row, col: fun(ind[row], ind[col])
        
        # Compute full matrices
        for i in range(len(self.row1)):
            # Cluster indices
            indr = tree.cind[self.row1[i], :]
            indc = tree.cind[self.col1[i], :]
            
            # Rows and columns
            row_range = np.arange(indr[0], indr[1] + 1)
            col_range = np.arange(indc[0], indc[1] + 1)
            row, col = np.meshgrid(row_range, col_range, indexing='ij')
            
            # Get function values
            self.val[i] = fun2(row.ravel(), col.ravel()).reshape(row.shape)
        
        # For low-rank matrices, we would need ACA implementation
        # This is simplified - in practice would use adaptive cross approximation
        for i in range(len(self.row2)):
            # Cluster indices
            indr = tree.cind[self.row2[i], :]
            indc = tree.cind[self.col2[i], :]
            
            # Create temporary full matrix and decompose
            row_range = np.arange(indr[0], indr[1] + 1)
            col_range = np.arange(indc[0], indc[1] + 1)
            row, col = np.meshgrid(row_range, col_range, indexing='ij')
            
            temp_mat = fun2(row.ravel(), col.ravel()).reshape(row.shape)
            
            # Simple SVD-based low-rank approximation
            U, s, Vt = svd(temp_mat, full_matrices=False)
            
            # Truncate based on tolerance
            k = np.sum(np.cumsum(s) / np.sum(s) <= 1 - self.htol)
            k = min(k, self.kmax, len(s))
            
            if k > 0:
                self.lhs[i] = U[:, :k] * np.sqrt(s[:k])
                self.rhs[i] = Vt[:k, :].T * np.sqrt(s[:k])
            else:
                self.lhs[i] = np.zeros((row.shape[0], 1))
                self.rhs[i] = np.zeros((row.shape[1], 1))
        
        return self
    
    def fillval(self, fun):
        """
        Fill matrix with user-defined function.
        
        Parameters
        ----------
        fun : function
            User-defined function fun(row, col)
            
        Returns
        -------
        HMatrix
            H-matrix with full matrices
        """
        tree = self.tree
        ind = tree.ind[:, 0]  # transformation to cluster indices
        
        # Modify input function
        fun2 = lambda row, col: fun(ind[row], ind[col])
        
        # Compute full matrices
        for i in range(len(self.row1)):
            # Cluster indices
            indr = tree.cind[self.row1[i], :]
            indc = tree.cind[self.col1[i], :]
            
            # Rows and columns
            row_range = np.arange(indr[0], indr[1] + 1)
            col_range = np.arange(indc[0], indc[1] + 1)
            row, col = np.meshgrid(row_range, col_range, indexing='ij')
            
            # Get function values
            self.val[i] = fun2(row.ravel(), col.ravel()).reshape(row.shape)
        
        return self
    
    def full(self):
        """
        Compute full matrix from hierarchical matrix.
        
        Returns
        -------
        ndarray
            Full matrix
        """
        # Get matrix size
        n = self.tree.ind.shape[0]
        mat = np.zeros((n, n))
        
        # Add full matrices
        for i in range(len(self.row1)):
            if self.val[i] is not None:
                indr = self.tree.cind[self.row1[i], :]
                indc = self.tree.cind[self.col1[i], :]
                row_slice = slice(indr[0], indr[1] + 1)
                col_slice = slice(indc[0], indc[1] + 1)
                mat[row_slice, col_slice] = self.val[i]
        
        # Add low-rank matrices
        for i in range(len(self.row2)):
            if self.lhs[i] is not None and self.rhs[i] is not None:
                indr = self.tree.cind[self.row2[i], :]
                indc = self.tree.cind[self.col2[i], :]
                row_slice = slice(indr[0], indr[1] + 1)
                col_slice = slice(indc[0], indc[1] + 1)
                mat[row_slice, col_slice] = self.lhs[i] @ self.rhs[i].T
        
        # Transform to particle indices
        ind = self.tree.ind[:, 1]
        return mat[np.ix_(ind, ind)]
    
    def compression(self):
        """
        Degree of compression for H-matrix.
        
        Returns
        -------
        float
            Ratio between elements of H-matrix and full matrix
        """
        # Number of H-matrix elements
        n_full = sum(val.size if val is not None else 0 for val in self.val)
        n_lr = sum((lhs.size if lhs is not None else 0) + 
                  (rhs.size if rhs is not None else 0) 
                  for lhs, rhs in zip(self.lhs, self.rhs))
        
        n_total = n_full + n_lr
        
        # Matrix size
        matrix_size = self.tree.matsize(self.tree)
        n_full_matrix = matrix_size[0] * matrix_size[1]
        
        return n_total / n_full_matrix
    
    def diag(self):
        """
        Diagonal of hierarchical matrix.
        
        Returns
        -------
        ndarray
            Diagonal elements
        """
        tree = self.tree
        d = np.zeros(tree.ind.shape[0])
        
        # Find matrices on diagonal
        diagonal_indices = np.where(self.row1 == self.col1)[0]
        
        # Loop over sub-matrices
        for i in diagonal_indices:
            ind_range = slice(tree.cind[self.row1[i], 0], tree.cind[self.col1[i], 1] + 1)
            d[ind_range] = np.diag(self.val[i])
        
        # Convert to particle indices
        return d[tree.ind[:, 1]]
    
    def eye(self):
        """
        Hierarchical unit matrix.
        
        Returns
        -------
        HMatrix
            Unit H-matrix
        """
        # Find diagonal matrices
        diagonal_mask = self.row1 == self.col1
        
        # Clear matrices
        self.val = [None] * len(self.row1)
        self.lhs = [None] * len(self.row2)
        self.rhs = [None] * len(self.row2)
        
        # Pad with zeros
        self.pad()
        
        # Set diagonal matrices
        for i in range(len(self.row1)):
            if diagonal_mask[i]:
                self.val[i] = np.eye(self.val[i].shape[0])
        
        return self
    
    def pad(self):
        """
        Pad missing submatrices of H-matrix with zeros.
        
        Returns
        -------
        HMatrix
            Padded H-matrix
        """
        tree = self.tree
        siz = tree.cind[:, 1] - tree.cind[:, 0] + 1
        
        # Pad full matrices
        for i in range(len(self.row1)):
            if self.val[i] is None:
                m, n = siz[self.row1[i]], siz[self.col1[i]]
                self.val[i] = np.zeros((m, n))
        
        # Pad low-rank matrices
        for i in range(len(self.row2)):
            if self.lhs[i] is None:
                self.lhs[i] = np.zeros((siz[self.row2[i]], 1))
            if self.rhs[i] is None:
                self.rhs[i] = np.zeros((siz[self.col2[i]], 1))
        
        return self
    
    def truncate(self, htol=None):
        """
        Truncate hierarchical matrix.
        
        Parameters
        ----------
        htol : float, optional
            Tolerance for truncation of low-rank matrices
            
        Returns
        -------
        HMatrix
            Truncated H-matrix
        """
        if htol is None:
            htol = self.htol
        
        for i in range(len(self.lhs)):
            if self.lhs[i] is not None and self.rhs[i] is not None:
                self.lhs[i], self.rhs[i] = self._truncate_lr(self.lhs[i], self.rhs[i], htol)
        
        self.htol = htol
        return self
    
    def _truncate_lr(self, lhs, rhs, htol):
        """Truncate low-rank matrix."""
        # Deal with zero matrices
        if np.linalg.norm(lhs) < np.finfo(float).eps or np.linalg.norm(rhs) < np.finfo(float).eps:
            return lhs, rhs
        
        q1, r1 = qr(lhs, mode='economic')
        q2, r2 = qr(rhs, mode='economic')
        
        # SVD decomposition
        u, s, vt = svd(r1 @ r2.T)
        
        # Find largest singular values
        s_cumsum = np.cumsum(s)
        k = np.sum(s_cumsum < (1 - htol) * s_cumsum[-1])
        
        if k > 0:
            # Truncate low-rank matrices
            lhs_new = q1[:, :k] @ u[:, :k] @ np.diag(s[:k])
            rhs_new = q2[:, :k] @ vt[:k, :].T
            return lhs_new, rhs_new.conj()
        else:
            return np.zeros_like(lhs[:, :1]), np.zeros_like(rhs[:, :1])
    
    def __add__(self, other):
        """Add hierarchical matrices."""
        if issparse(other):
            return self._plus_sparse(other)
        elif isinstance(other, HMatrix):
            return self._plus_hmatrix(other)
        else:
            raise NotImplementedError("Addition not implemented for this type")
    
    def __sub__(self, other):
        """Subtract hierarchical matrices."""
        return self + (-other)
    
    def __neg__(self):
        """Unary minus for hierarchical matrix."""
        result = HMatrix()
        result.__dict__ = self.__dict__.copy()
        result.val = [-val if val is not None else None for val in self.val]
        result.lhs = [-lhs if lhs is not None else None for lhs in self.lhs]
        return result
    
    def __mul__(self, other):
        """Multiplication of hierarchical matrices."""
        if np.isscalar(other):
            result = HMatrix()
            result.__dict__ = self.__dict__.copy()
            result.val = [other * val if val is not None else None for val in self.val]
            result.lhs = [other * lhs if lhs is not None else None for lhs in self.lhs]
            return result
        else:
            raise NotImplementedError("Multiplication not fully implemented")
    
    def _plus_sparse(self, sparse_mat):
        """Add sparse matrix to H-matrix."""
        # This would implement sparse matrix addition
        raise NotImplementedError("Sparse matrix addition not implemented")
    
    def _plus_hmatrix(self, other):
        """Add two H-matrices."""
        # This would implement H-matrix addition
        raise NotImplementedError("H-matrix addition not implemented")
    
    def plotrank(self):
        """Plot rank of hierarchical matrix."""
        # Allocate array
        matrix_size = self.tree.matsize(self.tree)
        mat = np.zeros(matrix_size, dtype=np.uint16)
        
        # Fill with ranks
        for i in range(len(self.row2)):
            if self.lhs[i] is not None:
                indr = self.tree.cind[self.row2[i], :]
                indc = self.tree.cind[self.col2[i], :]
                row_slice = slice(indr[0], indr[1] + 1)
                col_slice = slice(indc[0], indc[1] + 1)
                mat[row_slice, col_slice] = self.lhs[i].shape[1]
        
        # Plot
        plt.figure()
        plt.imshow(mat, cmap='viridis')
        plt.colorbar(label='Rank')
        plt.title('H-matrix Rank Structure')
        plt.show()
        
        return mat