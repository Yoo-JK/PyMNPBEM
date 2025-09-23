import numpy as np
from scipy.sparse import csr_matrix, diags


class GreenRet:
    """
    Green functions for solution of full Maxwell equations.
    """
    
    def __init__(self, p1, p2, **options):
        """
        Initialize Green function for solution of full Maxwell equations.
        
        Parameters:
        -----------
        p1 : object
            Green function between points p1 and comparticle p2
        p2 : object
            Green function between points p1 and comparticle p2
        **options : dict
            Options for calculation of Green function
        """
        self.p1 = p1
        self.p2 = p2
        self.op = options
        
        # Private properties
        self.deriv = options.get('deriv', 'cart')  # 'cart' or 'norm'
        self.order = options.get('order', 2)
        self.ind = np.array([])     # index to face elements with refinement
        self.g = np.array([])       # refined elements for Green function
        self.f = np.array([])       # refined elements for derivative of Green function
        
        self._init()
    
    def _init(self):
        """Initialize Green function."""
        # Import refinematrix from green module (placeholder)
        # from ..green.refinematrix import refinematrix
        
        # For now, skip refinement initialization
        # ir = refinematrix(self.p1, self.p2, **self.op)
        # self.ind = np.where(ir.toarray().ravel() != 0)[0]
        
        print(f"GreenRet initialized: p1.n={getattr(self.p1, 'n', '?')}, p2.n={getattr(self.p2, 'n', '?')}")
    
    def __str__(self):
        """Command window display."""
        return f"greenret: p1={self.p1}, p2={self.p2}, op={self.op}"
    
    def __getattr__(self, name):
        """Handle property access for G, F, H1, H2, Gp."""
        if name in ['G', 'F', 'H1', 'H2', 'Gp', 'H1p', 'H2p']:
            return lambda k, *args: self.eval(k, name, *args)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def eval(self, k_or_ind, *args):
        """
        Evaluate Green function.
        
        Parameters:
        -----------
        k_or_ind : float or array_like
            Wavenumber k, or indices for matrix elements
        *args : 
            Additional arguments (keys like 'G', 'F', etc.)
        """
        if isinstance(k_or_ind, (int, np.integer)) or \
           (isinstance(k_or_ind, np.ndarray) and k_or_ind.dtype in [np.int32, np.int64]):
            # Called with indices: eval(obj, ind, k, key1, key2, ...)
            return self._eval_indexed(k_or_ind, *args)
        else:
            # Called with wavenumber: eval(obj, k, key1, key2, ...)
            return self._eval_full(k_or_ind, *args)
    
    def _eval_full(self, k, *keys):
        """Evaluate Green function for all matrix elements."""
        if not hasattr(self.p1, 'pos') or not hasattr(self.p2, 'pos'):
            print("Warning: p1 or p2 missing pos attribute")
            return np.zeros((getattr(self.p1, 'n', 1), getattr(self.p2, 'n', 1)))
        
        pos1, pos2 = self.p1.pos, self.p2.pos
        n1, n2 = pos1.shape[0], pos2.shape[0]
        
        # Calculate distance matrix
        x = pos1[:, 0:1] - pos2[:, 0].T
        y = pos1[:, 1:2] - pos2[:, 1].T  
        z = pos1[:, 2:3] - pos2[:, 2].T
        d = np.maximum(np.sqrt(x**2 + y**2 + z**2), np.finfo(float).eps)
        
        # Area weighting
        area = getattr(self.p2, 'area', np.ones(n2))
        if hasattr(area, '__len__'):
            area_matrix = diags(area, shape=(n2, n2))
        else:
            area_matrix = diags([area] * n2, shape=(n2, n2))
        
        results = []
        for key in keys:
            if key == 'G':
                # Green function
                G = (1.0 / d) @ area_matrix
                G = G * np.exp(1j * k * d)
                results.append(G)
                
            elif key in ['F', 'H1', 'H2']:
                # Surface derivative of Green function
                if self.deriv == 'norm' and hasattr(self.p1, 'nvec'):
                    # Normal derivative
                    nvec = self.p1.nvec
                    in_prod = (x * nvec[:, 0:1] + 
                              y * nvec[:, 1:2] + 
                              z * nvec[:, 2:3])
                    F = (in_prod * (1j * k - 1.0 / d) / d**2) @ area_matrix
                    F = F * np.exp(1j * k * d)
                else:
                    # Simplified calculation
                    F = ((1j * k - 1.0 / d) / d**2) @ area_matrix
                    F = F * np.exp(1j * k * d)
                
                if key == 'F':
                    results.append(F)
                elif key == 'H1':
                    diag_correction = 2 * np.pi * (d == np.finfo(float).eps)
                    results.append(F + diag_correction)
                elif key == 'H2':
                    diag_correction = 2 * np.pi * (d == np.finfo(float).eps)
                    results.append(F - diag_correction)
                    
            elif key in ['Gp', 'H1p', 'H2p']:
                # Derivative of Green function (placeholder)
                if self.deriv == 'cart':
                    f = (1j * k - 1.0 / d) / d**2
                    Gp = np.stack([
                        (f * x) @ area_matrix,
                        (f * y) @ area_matrix,
                        (f * z) @ area_matrix
                    ], axis=1)
                    Gp = Gp * np.exp(1j * k * d)[:, np.newaxis]
                    
                    if key == 'Gp':
                        results.append(Gp)
                    elif key in ['H1p', 'H2p']:
                        # Add/subtract normal correction
                        if hasattr(self.p1, 'nvec'):
                            diag_mask = (d == np.finfo(float).eps)
                            correction = 2 * np.pi * self.p1.nvec[:, np.newaxis, :] * diag_mask[:, np.newaxis, np.newaxis]
                            if key == 'H1p':
                                results.append(Gp + correction)
                            else:
                                results.append(Gp - correction)
                        else:
                            results.append(Gp)
                else:
                    raise ValueError("Gp derivatives require deriv='cart'")
            else:
                raise ValueError(f"Unknown key: {key}")
        
        return results[0] if len(results) == 1 else results
    
    def _eval_indexed(self, ind, k, *keys):
        """Evaluate Green function for specific matrix elements."""
        # Convert linear indices to row, col
        n1 = getattr(self.p1, 'n', self.p1.pos.shape[0])
        n2 = getattr(self.p2, 'n', self.p2.pos.shape[0]) 
        row, col = np.unravel_index(ind, (n1, n2))
        
        pos1 = self.p1.pos[row, :]
        pos2 = self.p2.pos[col, :]
        
        # Distance
        d = np.maximum(np.sqrt(np.sum((pos1 - pos2)**2, axis=1)), np.finfo(float).eps)
        
        # Area
        area = getattr(self.p2, 'area', np.ones(len(col)))
        if hasattr(area, '__len__'):
            area = area[col]
        
        results = []
        for key in keys:
            if key == 'G':
                G = (1.0 / d) * area * np.exp(1j * k * d)
                results.append(G)
            elif key in ['F', 'H1', 'H2']:
                if self.deriv == 'norm' and hasattr(self.p1, 'nvec'):
                    in_prod = np.sum(self.p1.nvec[row, :] * (pos1 - pos2), axis=1)
                    F = in_prod * (1j * k - 1.0 / d) / d**2 * area
                else:
                    F = (1j * k - 1.0 / d) / d**2 * area
                F = F * np.exp(1j * k * d)
                
                if key == 'F':
                    results.append(F)
                elif key == 'H1':
                    results.append(F + 2 * np.pi * (d == np.finfo(float).eps))
                elif key == 'H2':
                    results.append(F - 2 * np.pi * (d == np.finfo(float).eps))
            else:
                # Placeholder for other keys
                results.append(np.zeros(len(ind)))
        
        return results[0] if len(results) == 1 else results
    
    def diag(self, ind, f):
        """Set diagonal elements of surface derivative of Green function."""
        ind = np.ravel_multi_index((ind, ind), (self.p1.n, self.p1.n))
        
        if len(self.ind) == 0:
            self.ind = ind
            self.g = np.zeros(len(ind))
            self.f = f
        else:
            # Update existing diagonal elements
            for i, idx in enumerate(ind):
                mask = self.ind == idx
                if f.ndim == 1:
                    self.f[mask] += f[i]
                else:
                    self.f[mask, :] += f[i, :]
        
        return self
    
    def mat2cell(self, p1_cell, p2_cell):
        """Split Green function into cell array."""
        # Get dimensions
        siz1 = [p.n for p in p1_cell]
        siz2 = [p.n for p in p2_cell] 
        
        # Create cell array of Green functions
        result = []
        for i in range(len(p1_cell)):
            row = []
            for j in range(len(p2_cell)):
                # Create new Green function for each cell
                new_green = GreenRet(p1_cell[i], p2_cell[j], **self.op)
                row.append(new_green)
            result.append(row)
        
        return result


# Alias for backward compatibility
greenret = GreenRet