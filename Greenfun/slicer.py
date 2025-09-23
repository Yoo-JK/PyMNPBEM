import numpy as np
from .bemoptions import get_bemoptions
from .misc import misc


class Slicer:
    """
    Successively work through large matrices.
    Given the size and indices for a possibly large matrix, this class
    provides the functions to successively work through the 'slices' of
    the matrix.
    
    Usage:
        # Set up slicer
        s = Slicer(siz)
        # Work through slices
        for i in range(s.n):
            # Indices, rows, and columns of slice
            ind, row, col = s(i)
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize slicer object.
        
        Parameters
        ----------
        siz : tuple
            Size of matrix
        ind1 : array_like, optional
            Row indices
        ind2 : array_like, optional
            Column indices
        *args, **kwargs : optional
            Property pairs
            
        Properties
        ----------
        memsize : int, optional
            Maximum memory size
        """
        self.siz = None
        self.ind1 = None
        self.ind2 = None
        self.n = None
        self.sind2 = None
        
        self._init(*args, **kwargs)
    
    def __str__(self):
        """Command window display."""
        info = {
            'siz': self.siz,
            'ind1': self.ind1,
            'ind2': self.ind2
        }
        return f"Slicer:\n{info}"
    
    def __call__(self, i):
        """
        Compute indices for slice.
        
        Parameters
        ----------
        i : int
            Slice index
            
        Returns
        -------
        tuple
            ind : array_like
                Matrix indices for slice
            row : array_like
                Row indices for slice
            col : array_like
                Column indices for slice
        """
        return self.eval(i)
    
    def _init(self, siz, *args, **kwargs):
        """Initialize slicer object."""
        self.siz = siz
        
        # Deal with different calling sequences
        if args and not isinstance(args[0], str):
            self.ind1 = args[0]
            self.ind2 = args[1]
            remaining_args = args[2:]
        else:
            self.ind1 = np.arange(siz[0])
            self.ind2 = np.arange(siz[1])
            remaining_args = args
        
        # Get options
        op = get_bemoptions(*remaining_args, **kwargs)
        
        # Use default value for memory size
        if not hasattr(op, 'memsize'):
            op.memsize = misc.memsize()
        
        # Make sure that memsize is large enough
        assert op.memsize > siz[0], "Memory size must be larger than number of rows"
        
        # Table of slice indices
        slice_size = op.memsize // len(self.ind1)
        k = np.arange(0, len(self.ind2) + slice_size, slice_size)
        
        # Ensure we don't go beyond the actual size
        if k[-1] != len(self.ind2):
            k = np.append(k, len(self.ind2))
        
        # Save slice indices and number of slices
        self.sind2 = k
        self.n = len(k) - 1
    
    def eval(self, i):
        """
        Compute indices for slice.
        
        Parameters
        ----------
        i : int
            Slice index
            
        Returns
        -------
        tuple
            ind : array_like
                Matrix indices for slice
            row : array_like
                Row indices for slice
            col : array_like
                Column indices for slice
        """
        assert i < self.n, f"Slice index {i} must be less than number of slices {self.n}"
        
        # Row and column indices
        row = self.ind1
        col = self.ind2[self.sind2[i]:self.sind2[i + 1]]
        
        # Linear indices using meshgrid
        r, c = np.meshgrid(row, col, indexing='ij')
        
        # Convert to linear indices
        ind = np.ravel_multi_index((r.ravel(), c.ravel()), self.siz)
        
        return ind, row, col
    
    @property
    def num_slices(self):
        """Number of slices."""
        return self.n
    
    def get_slice_info(self, i):
        """
        Get information about a specific slice.
        
        Parameters
        ----------
        i : int
            Slice index
            
        Returns
        -------
        dict
            Information about the slice including size and memory usage
        """
        if i >= self.n:
            raise IndexError(f"Slice index {i} out of range")
        
        col_start = self.sind2[i]
        col_end = self.sind2[i + 1]
        slice_cols = col_end - col_start
        slice_size = len(self.ind1) * slice_cols
        
        return {
            'slice_index': i,
            'row_range': (0, len(self.ind1)),
            'col_range': (col_start, col_end),
            'num_rows': len(self.ind1),
            'num_cols': slice_cols,
            'total_elements': slice_size
        }
    
    def get_all_slice_info(self):
        """Get information about all slices."""
        return [self.get_slice_info(i) for i in range(self.n)]