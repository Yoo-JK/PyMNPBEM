import numpy as np
from collections import defaultdict


class BlockMatrix:
    """
    Base class for block matrix.
    Given a block matrix with submatrices of size siz1 x siz2, this
    class provides a conversion of indices from total to submatrices
    and vice versa.
    """
    
    def __init__(self, *args):
        """
        Initialize block matrix.
        
        Parameters
        ----------
        siz1 : array_like
            Row sizes of submatrices
        siz2 : array_like
            Column sizes of submatrices
        """
        self.siz1 = None
        self.siz2 = None
        self._n1 = None
        self._n2 = None
        self._row = None
        self._col = None
        self._siz = None
        
        if args:
            self._init(*args)
    
    def __str__(self):
        """Command window display."""
        info = {
            'siz1': self.siz1,
            'siz2': self.siz2
        }
        return f"BlockMatrix:\n{info}"
    
    def _init(self, siz1, siz2):
        """Initialize block matrix."""
        self.siz1 = np.array(siz1)
        self.siz2 = np.array(siz2)
        
        # Total size of block matrix
        self._n1 = np.sum(siz1)
        self._n2 = np.sum(siz2)
        
        # Initial rows and columns of submatrices
        r = np.concatenate([[0], np.cumsum(siz1[:-1])])
        c = np.concatenate([[0], np.cumsum(siz2[:-1])])
        
        # Convert to meshgrid for all combinations
        row_grid, col_grid = np.meshgrid(r, c, indexing='ij')
        
        # Store as arrays
        self._row = row_grid
        self._col = col_grid
        
        # Size of submatrices
        siz1_grid, siz2_grid = np.meshgrid(siz1, siz2, indexing='ij')
        self._siz = np.stack([siz1_grid, siz2_grid], axis=-1)
    
    def ind2sub(self, ind):
        """
        Convert total index to cell array of subindices.
        
        Parameters
        ----------
        ind : array_like
            Total index (linear indices)
            
        Returns
        -------
        tuple
            sub : dict
                Dictionary of subindices for each block
            ind_dict : dict
                Index dictionary to be used in accumarray
        """
        # Convert to row and column indices
        r, c = np.unravel_index(ind, (self._n1, self._n2))
        
        # Find sub-matrices for rows and columns
        row_bins = np.concatenate([[0], np.cumsum(self.siz1)])
        col_bins = np.concatenate([[0], np.cumsum(self.siz2)])
        
        indr = np.digitize(r, row_bins) - 1
        indc = np.digitize(c, col_bins) - 1
        
        # Ensure indices are within bounds
        indr = np.clip(indr, 0, len(self.siz1) - 1)
        indc = np.clip(indc, 0, len(self.siz2) - 1)
        
        # Group elements by block indices
        sub = defaultdict(list)
        ind_dict = defaultdict(list)
        
        for i, (block_r, block_c) in enumerate(zip(indr, indc)):
            key = (block_r, block_c)
            
            # Calculate relative position within block
            r0 = self._row[block_r, block_c]
            c0 = self._col[block_r, block_c]
            
            # Convert to sub-index within the block
            sub_r = r[i] - r0
            sub_c = c[i] - c0
            
            # Convert to linear index within block
            block_size = self._siz[block_r, block_c]
            sub_ind = np.ravel_multi_index([sub_r, sub_c], block_size)
            
            sub[key].append(sub_ind)
            ind_dict[key].append(i)
        
        # Convert lists to arrays
        for key in sub:
            sub[key] = np.array(sub[key])
            ind_dict[key] = np.array(ind_dict[key])
        
        return dict(sub), dict(ind_dict)
    
    def accumarray(self, ind_dict, val):
        """
        Assemble array together.
        
        Parameters
        ----------
        ind_dict : dict
            Index dictionary from previous call to ind2sub
        val : dict
            Dictionary with values for each block
            
        Returns
        -------
        ndarray
            Total value array
        """
        # Find total length needed
        total_length = 0
        for key in ind_dict:
            total_length = max(total_length, np.max(ind_dict[key]) + 1)
        
        # Initialize result array
        result = np.zeros(total_length)
        
        # Accumulate values
        for key in val:
            if key in ind_dict:
                indices = ind_dict[key]
                values = val[key]
                
                # Handle case where values might be arrays
                if np.isscalar(values):
                    result[indices] += values
                else:
                    # If values is an array, add element-wise
                    if len(values) == len(indices):
                        result[indices] += values
                    else:
                        # Broadcast if needed
                        result[indices] += values.ravel()[:len(indices)]
        
        return result
    
    def get_block_indices(self, block_row, block_col):
        """
        Get the global indices for a specific block.
        
        Parameters
        ----------
        block_row : int
            Block row index
        block_col : int
            Block column index
            
        Returns
        -------
        tuple
            Global row and column indices for the block
        """
        if block_row >= len(self.siz1) or block_col >= len(self.siz2):
            raise IndexError("Block indices out of range")
        
        # Starting positions
        r0 = self._row[block_row, block_col]
        c0 = self._col[block_row, block_col]
        
        # Block sizes
        nr = self.siz1[block_row]
        nc = self.siz2[block_col]
        
        # Global indices
        row_indices = np.arange(r0, r0 + nr)
        col_indices = np.arange(c0, c0 + nc)
        
        return row_indices, col_indices
    
    def get_block_slice(self, block_row, block_col):
        """
        Get slice objects for a specific block.
        
        Parameters
        ----------
        block_row : int
            Block row index
        block_col : int
            Block column index
            
        Returns
        -------
        tuple
            Row and column slice objects
        """
        row_indices, col_indices = self.get_block_indices(block_row, block_col)
        return slice(row_indices[0], row_indices[-1] + 1), slice(col_indices[0], col_indices[-1] + 1)
    
    @property
    def shape(self):
        """Total matrix shape."""
        return (self._n1, self._n2)
    
    @property
    def num_blocks(self):
        """Number of blocks."""
        return (len(self.siz1), len(self.siz2))