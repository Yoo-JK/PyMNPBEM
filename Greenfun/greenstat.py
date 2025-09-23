import numpy as np
from scipy.sparse import diags


class GreenStat:
    """
    Green functions for quasistatic approximation.
    
    This class implements Green functions in the quasistatic limit,
    where the wavelength is much larger than the particle size.
    """
    
    def __init__(self, p1, p2, **options):
        """
        Initialize Green function in quasistatic approximation.
        
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
        self.ind = np.array([])     # index to face elements with refinement
        self.g = np.array([])       # refined elements for Green function
        self.f = np.array([])       # refined elements for derivative of Green function
        
        self._init(**options)
    
    def _init(self, **options):
        """Initialize Green function."""
        # Import refinematrix from green module (placeholder)
        # from ..green.refinematrix import refinematrix
        
        # For now, skip refinement initialization
        # ir = refinematrix(self.p1, self.p2, **options)
        # self.ind = np.where(ir.toarray().ravel() != 0)[0]
        
        print(f"GreenStat initialized: p1.n={getattr(self.p1, 'n', '?')}, p2.n={getattr(self.p2, 'n', '?')}")
    
    def __str__(self):
        """Command window display."""
        return f"greenstat: p1={self.p1}, p2={self.p2}, op={self.op}"
    
    def __getattr__(self, name):
        """Handle property access for G, F, H1, H2, Gp."""
        if name in ['G', 'F', 'H1', 'H2', 'Gp', 'H1p', 'H2p']:
            return lambda *args: self.eval(name, *args)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def diag(self, ind, f):
        """
        Set diagonal elements of surface derivative of Green function.
        
        Parameters:
        -----------
        ind : array_like
            Index to diagonal elements
        f : array_like
            Value of diagonal element
        """
        # Convert indices to linear indices
        ind = np.ravel_multi_index((ind, ind), (self.p1.n, self.p1.n))
        
        if len(self.ind) == 0:
            # Save indices and values
            self.ind = ind
            self.g = np.zeros(len(ind))
            self.f = f
        else:
            # Update existing diagonal elements
            _, i1, i2 = np.intersect1d(self.ind, ind, return_indices=True)
            if f.ndim == 1:
                self.f[i1] += f[i2]
            else:
                self.f[i1, :] += f[i2, :]
        
        return self
    
    def eval(self, *args):
        """
        Evaluate Green function.
        
        Parameters:
        -----------
        *args : 
            Arguments can be:
            - (key1, key2, ...) for full evaluation
            - (ind, key1, key2, ...) for indexed evaluation
        """
        if isinstance(args[0], str):
            # Called as eval(obj, key1, key2, ...)
            return self._eval_full(*args)
        else:
            # Called as eval(obj, ind, key1, key2, ...)
            return self._eval_indexed(*args)
    
    def _eval_full(self, *keys):
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
                # Green function: G = 1/r
                G = (1.0 / d) @ area_matrix
                # Apply refinement
                if len(self.ind) > 0:
                    G.flat[self.ind] = self.g
                results.append(G)
                
            elif key in ['F', 'H1', 'H2']:
                # Surface derivative of Green function
                if self.deriv == 'norm' and hasattr(self.p1, 'nvec'):
                    # Normal derivative only
                    nvec = self.p1.nvec
                    in_prod = (x * nvec[:, 0:1] + 
                              y * nvec[:, 1:2] + 
                              z * nvec[:, 2:3])
                    F = -(in_prod / d**3) @ area_matrix
                    if len(self.ind) > 0:
                        F.flat[self.ind] = self.f
                else:
                    # Simplified calculation
                    F = -(1.0 / d**3) @ area_matrix
                    if len(self.ind) > 0:
                        F.flat[self.ind] = self.f
                
                if key == 'F':
                    results.append(F)
                elif key == 'H1':
                    diag_correction = 2 * np.pi * (d == np.finfo(float).eps)
                    results.append(F + diag_correction)
                elif key == 'H2':
                    diag_correction = 2 * np.pi * (d == np.finfo(float).eps)
                    results.append(F - diag_correction)
                    
            elif key in ['Gp', 'H1p', 'H2p']:
                # Derivative of Green function
                if self.deriv == 'cart':
                    # Cartesian derivatives: Gp = -grad(1/r) = r/r^3
                    Gp = np.stack([
                        -(x / d**3) @ area_matrix,
                        -(y / d**3) @ area_matrix,
                        -(z / d**3) @ area_matrix
                    ], axis=1)  # Shape: (n1, 3, n2)
                    
                    # Apply refinement
                    if len(self.ind) > 0 and self.f.ndim > 1:
                        for i in range(3):
                            Gp[:, i, :].flat[self.ind] = self.f[:, i]
                    
                    if key == 'Gp':
                        results.append(Gp)
                    elif key in ['H1p', 'H2p']:
                        # Add/subtract normal correction
                        if hasattr(self.p1, 'nvec'):
                            diag_mask = (d == np.finfo(float).eps)
                            correction = 2 * np.pi * self.p1.nvec[:, :, np.newaxis] * diag_mask[np.newaxis, np.newaxis, :]
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
    
    def _eval_indexed(self, ind, *keys):
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
        
        # Find refined elements
        _, ind1, ind2 = np.intersect1d(self.ind, ind, return_indices=True)
        
        results = []
        for key in keys:
            if key == 'G':
                G = (1.0 / d) * area
                if len(ind1) > 0:
                    G[ind2] = self.g[ind1]
                results.append(G)
                
            elif key in ['F', 'H1', 'H2']:
                if self.deriv == 'norm' and hasattr(self.p1, 'nvec'):
                    # Normal derivative
                    in_prod = np.sum(self.p1.nvec[row, :] * (pos1 - pos2), axis=1)
                    F = -in_prod / d**3 * area
                else:
                    # Simplified
                    F = -1.0 / d**3 * area
                
                if len(ind1) > 0:
                    if self.f.ndim == 1:
                        F[ind2] = self.f[ind1]
                    else:
                        F[ind2] = np.sum(self.p1.nvec[row[ind2], :] * self.f[ind1, :], axis=1)
                
                if key == 'F':
                    results.append(F)
                elif key == 'H1':
                    results.append(F + 2 * np.pi * (d == np.finfo(float).eps))
                elif key == 'H2':
                    results.append(F - 2 * np.pi * (d == np.finfo(float).eps))
                    
            elif key in ['Gp', 'H1p', 'H2p']:
                if self.deriv == 'cart':
                    # Cartesian derivatives
                    Gp = np.column_stack([
                        -(pos1[:, 0] - pos2[:, 0]) / d**3 * area,
                        -(pos1[:, 1] - pos2[:, 1]) / d**3 * area,
                        -(pos1[:, 2] - pos2[:, 2]) / d**3 * area
                    ])
                    
                    if len(ind1) > 0:
                        Gp[ind2, 0] = self.f[ind1, 0]
                        Gp[ind2, 1] = self.f[ind1, 1] 
                        Gp[ind2, 2] = self.f[ind1, 2]
                    
                    if key == 'Gp':
                        results.append(Gp)
                    elif key in ['H1p', 'H2p']:
                        # Add normal correction
                        if hasattr(self.p1, 'nvec'):
                            diag_mask = (d == np.finfo(float).eps)
                            correction = 2 * np.pi * self.p1.nvec[row, :] * diag_mask[:, np.newaxis]
                            if key == 'H1p':
                                results.append(Gp + correction)
                            else:
                                results.append(Gp - correction)
                        else:
                            results.append(Gp)
                else:
                    raise ValueError("Gp derivatives require deriv='cart'")
            else:
                results.append(np.zeros(len(ind)))
        
        return results[0] if len(results) == 1 else results


# Alias for backward compatibility
greenstat = GreenStat