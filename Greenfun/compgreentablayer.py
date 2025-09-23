import numpy as np


class CompGreenTabLayer:
    """
    Green function for layer structure and different media.
    
    COMPGREENTABLAYER computes the reflected Green function and its
    derivatives for a layer structure, and saves the results for a
    rectangular mesh of radial and height values. The stored values can
    then be used for interpolation with arbitrary radial and height values.
    """
    
    def __init__(self, layer, *tabs):
        """
        Initialize compound Green function object for layer structure.
        
        Parameters:
        -----------
        layer : object
            Layer structure
        *tabs : objects
            Grids for tabulated r and z-values
        """
        self.layer = None  # layer structure
        self.g = []        # cell array of tabulated Green functions
        
        self._init(layer, *tabs)
    
    def _init(self, layer, *tabs):
        """Initialize compound Green function object for layer structure."""
        # Handle case where tab is passed as a single list/array
        if len(tabs) == 1 and hasattr(tabs[0], '__iter__') and not hasattr(tabs[0], 'r'):
            tabs = tabs[0]
        
        # Save layer structure and allocate list for Green functions
        self.layer = layer
        self.g = []
        
        # Initialize Green function objects
        for tab in tabs:
            # Import from current module - would need actual greentablayer
            # from .greentablayer import GreenTabLayer
            # self.g.append(GreenTabLayer(layer, tab))
            
            # Placeholder for now
            self.g.append(self._create_greentablayer(layer, tab))
    
    def _create_greentablayer(self, layer, tab):
        """Create a GreenTabLayer object (placeholder)."""
        # This would create an actual GreenTabLayer object
        class PlaceholderGreenTabLayer:
            def __init__(self, layer, tab):
                self.layer = layer
                self.tab = tab
                self.G = None
                self.Fr = None
                self.Fz = None
                
            def eval(self, enei):
                self.G = {'p': np.ones((10, 10), dtype=complex)}
                self.Fr = {'p': np.ones((10, 10), dtype=complex)}
                self.Fz = {'p': np.ones((10, 10), dtype=complex)}
                return self
                
            def inside(self, r, z1, z2=None):
                return np.ones_like(r, dtype=bool)
                
            def __call__(self, r, z1, z2=None):
                G = {'p': np.ones_like(r, dtype=complex)}
                Fr = {'p': np.ones_like(r, dtype=complex)}
                Fz = {'p': np.ones_like(r, dtype=complex)}
                return G, Fr, Fz, r, np.abs(z1)
                
            def ismember(self, layer, enei=None):
                return True
                
            def set(self, enei, **options):
                return self
                
            def parset(self, enei, **options):
                return self
                
            @property
            def numel(self):
                return 100
        
        return PlaceholderGreenTabLayer(layer, tab)
    
    def __str__(self):
        """Command window display."""
        return f"compgreentablayer: layer={self.layer}, g={len(self.g)} tables"
    
    def eval(self, enei):
        """
        Evaluate table of Green functions for given wavelength.
        
        Parameters:
        -----------
        enei : float
            Wavelength of light in vacuum
            
        Returns:
        --------
        self : CompGreenTabLayer
            Evaluated Green function table
        """
        # Evaluate all Green function tables
        self.g = [g.eval(enei) for g in self.g]
        return self
    
    def inside(self, r, z1, z2=None):
        """
        Determine whether coordinates are inside of tabulated range.
        
        Parameters:
        -----------
        r : array_like
            Radii
        z1 : array_like
            Z-values
        z2 : array_like, optional
            Second z-values
            
        Returns:
        --------
        ind : array_like
            Index to Green function to be used for interpolation
        """
        # Evaluate inside function for all Green function objects
        if z2 is None:
            inside_results = [g.inside(r, z1).ravel() for g in self.g]
        else:
            inside_results = [g.inside(r, z1, z2).ravel() for g in self.g]
        
        # Stack results and find nonzero elements
        inside_matrix = np.column_stack(inside_results)
        rows, cols = np.where(inside_matrix)
        
        # Create index array
        ind = np.zeros(inside_matrix.shape[0], dtype=int)
        if len(rows) > 0:
            ind[rows] = cols + 1  # 1-based indexing like MATLAB
        
        return ind
    
    def interp(self, r, z1, z2=None):
        """
        Interpolate tabulated Green functions.
        
        Parameters:
        -----------
        r : array_like
            Radii
        z1 : array_like
            Z-distances
        z2 : array_like, optional
            Second z-distances
            
        Returns:
        --------
        G, Fr, Fz : dict
            Green function components
        ri, zi : array_like
            Radius and z-distance as used in interpolation
        """
        # Convert inputs to arrays
        r = np.asarray(r)
        z1 = np.asarray(z1)
        if z2 is not None:
            z2 = np.asarray(z2)
        
        # Index to Green function to be used for interpolation
        ind = self.inside(r, z1, z2)
        
        # Make sure all interpolation is possible
        assert not np.any(ind == 0), "Some points cannot be interpolated"
        
        # Get field names from first Green function
        if self.g[0].G is not None:
            names = list(self.g[0].G.keys())
        else:
            names = ['p']  # Default field name
        
        # Allocate arrays
        G, Fr, Fz = {}, {}, {}
        for name in names:
            G[name] = np.zeros_like(r, dtype=complex)
            Fr[name] = np.zeros_like(r, dtype=complex)
            Fz[name] = np.zeros_like(r, dtype=complex)
        
        # Radii and minimal z-distance as used in interpolation
        ri = np.zeros_like(r, dtype=float)
        zi = np.zeros_like(r, dtype=float)
        
        # Loop over unique Green function indices
        for i in np.unique(ind):
            if i == 0:
                continue
                
            # Find points using this Green function (convert to 0-based)
            mask = (ind == i)
            
            # Evaluate Green function
            if z2 is not None:
                g, fr, fz, ri_i, zi_i = self.g[i-1](r[mask], z1[mask], z2[mask])
            else:
                g, fr, fz, ri_i, zi_i = self.g[i-1](r[mask], z1[mask])
            
            # Save results
            ri[mask] = ri_i
            zi[mask] = zi_i
            
            for name in names:
                G[name][mask] = g[name]
                Fr[name][mask] = fr[name]
                Fz[name][mask] = fz[name]
        
        return G, Fr, Fz, ri, zi
    
    def ismember(self, layer, *args):
        """
        Determine whether precomputed table is compatible with input.
        
        Parameters:
        -----------
        layer : object
            Layer structure
        *args : 
            Additional arguments (enei, p, pt)
            
        Returns:
        --------
        is_compatible : bool
            True if compatible with precomputed Green functions
        """
        # Handle calling sequences with and without wavelengths
        if args and isinstance(args[0], (int, float, list, np.ndarray)):
            enei = args[0]
            is_results = [g.ismember(layer, enei) for g in self.g]
            remaining_args = args[1:]
        else:
            is_results = [g.ismember(layer) for g in self.g]
            remaining_args = args
        
        # Return False if any layer structure is not identical
        if not all(is_results):
            return False
        
        # Deal with additional position arguments
        if remaining_args:
            # Extract positions from particle/point objects
            positions = []
            
            for arg in remaining_args:
                if not isinstance(arg, list):
                    arg = [arg]
                
                pos_list = []
                for obj in arg:
                    # Try to get vertices for particles, fallback to pos
                    try:
                        pos_list.append(obj.verts)
                    except AttributeError:
                        pos_list.append(obj.pos)
                
                # Concatenate all positions
                positions.append(np.vstack(pos_list))
            
            # Get positions
            pos1 = positions[0]
            pos2 = positions[0]  # Use same for now
            
            # Add additional positions if provided
            if len(positions) == 2:
                pos1 = np.vstack([pos1, positions[1]])
            
            # Calculate distances
            from scipy.spatial.distance import cdist
            r = cdist(pos1[:, :2], pos2[:, :2])
            
            # Expand z-values
            z1 = np.repeat(pos1[:, 2:3], pos2.shape[0], axis=1)
            z2 = np.repeat(pos2[:, 2:3].T, pos1.shape[0], axis=0)
            
            # Check if all points can be interpolated
            ind = self.inside(r.ravel(), z1.ravel(), z2.ravel())
            return not np.any(ind == 0)
        
        return True
    
    def set(self, enei, **options):
        """
        Precompute Green function table for given wavelengths.
        
        Parameters:
        -----------
        enei : array_like
            Wavelengths of light in vacuum
        **options : dict
            Options including 'waitbar'
            
        Returns:
        --------
        self : CompGreenTabLayer
            Precomputed Green function table
        """
        # Number of elements for each Green function
        n_elements = [g.numel for g in self.g]
        
        # Range for waitbar limits
        cumsum_n = np.cumsum([0] + n_elements)
        limits = cumsum_n / cumsum_n[-1]
        
        # Loop over Green functions
        for i, g in enumerate(self.g):
            # Set waitbar limits for this Green function
            waitbar_limits = [limits[i], limits[i+1]]
            options_with_limits = options.copy()
            options_with_limits['waitbarlimits'] = waitbar_limits
            
            # Precompute Green function table
            self.g[i] = g.set(enei, **options_with_limits)
        
        return self
    
    def parset(self, enei, **options):
        """
        Same as set but with parallel computation.
        
        Parameters:
        -----------
        enei : array_like
            Wavelengths of light in vacuum
        **options : dict
            Options
            
        Returns:
        --------
        self : CompGreenTabLayer
            Precomputed Green function table
        """
        # Apply parset to all Green function objects
        self.g = [g.parset(enei, **options) for g in self.g]
        return self


# Alias for backward compatibility
compgreentablayer = CompGreenTabLayer