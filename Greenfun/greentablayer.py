import numpy as np
from scipy.interpolate import interp1d, griddata


class GreenTabLayer:
    """
    Green function for layer structure.
    
    GREENTABLAYER computes the reflected Green function and its
    derivatives for a layer structure, and saves the results for a
    rectangular mesh of radial and height values. The stored values can
    then be used for interpolation with arbitrary radial and height values.
    """
    
    def __init__(self, layer, tab):
        """
        Initialize Green function object for layer structure.
        
        Parameters:
        -----------
        layer : object
            Layer structure
        tab : object
            Grids for tabulated r and z-values
        """
        # Properties
        self.layer = None      # layer structure
        self.r = np.array([])  # radial values for tabulation
        self.z1 = np.array([]) # z-values for tabulation
        self.z2 = np.array([]) # z-values for tabulation
        self.pos = None        # structure with expanded r and z values
        self.G = None          # table for Green function
        self.Fr = None         # table for surface derivatives
        self.Fz = None         # table for surface derivatives
        
        # Precomputed data
        self.enei = np.array([])  # wavelengths for precomputed Green function
        self.Gsav = None          # precomputed Green function table
        self.Frsav = None         # precomputed surface derivatives
        self.Fzsav = None         # precomputed surface derivatives
        
        self._init(layer, tab)
    
    def _init(self, layer, tab):
        """Initialize Green function object for layer structure."""
        self.layer = layer
        
        # Save radii with minimum radius constraint
        self.r = np.unique(np.sort(np.maximum(layer.rmin, tab.r)))
        
        # Save z-values rounded to layer interfaces
        self.z1 = np.unique(np.sort(self._round_layer(layer, tab.z1)))
        self.z2 = np.unique(np.sort(self._round_layer(layer, tab.z2)))
        
        # Validate that all z-values are within one medium
        ind1 = np.unique(self._indlayer(layer, tab.z1))
        ind2 = np.unique(self._indlayer(layer, tab.z2))
        assert len(ind1) == 1, "All z1 values must be in same medium"
        assert len(ind2) == 1, "All z2 values must be in same medium"
    
    def __str__(self):
        """Command window display."""
        return (f"greentablayer: layer={self.layer}, r={self.r.shape}, "
                f"z1={self.z1.shape}, z2={self.z2.shape}")
    
    def __call__(self, *args):
        """Perform interpolation when called as obj(r, z1, z2)."""
        return self.interp(*args)
    
    @property
    def size(self):
        """Size of tabulated grid."""
        sizes = [len(self.r), len(self.z1), len(self.z2)]
        return [s for s in sizes if s != 1]  # Remove ones
    
    @property
    def numel(self):
        """Number of elements of tabulated grid."""
        return len(self.r) * len(self.z1) * len(self.z2)
    
    def eval(self, enei, key=None):
        """
        Evaluate table of Green functions for given wavelength.
        
        Parameters:
        -----------
        enei : float
            Wavelength of light in vacuum
        key : str, optional
            'new' to force recomputation
            
        Returns:
        --------
        self : GreenTabLayer
            Evaluated Green function table
        """
        # Compute Green functions if needed
        if self.Gsav is None or (key == 'new'):
            self.G, self.Fr, self.Fz, self.pos = self._compute_green_functions(
                self.layer, enei, self.r.ravel(), self.z1.T.ravel(), self.z2.ravel()
            )
        else:
            # Use precomputed Green functions
            assert (enei >= np.min(self.enei) and enei <= np.max(self.enei)), \
                "Energy outside precomputed range"
            
            # Get field names
            names = list(self.Gsav.keys())
            
            if len(self.enei) == 1:
                # Only single wavelength
                for name in names:
                    self.G[name] = np.squeeze(self.Gsav[name])
                    self.Fr[name] = np.squeeze(self.Frsav[name])
                    self.Fz[name] = np.squeeze(self.Fzsav[name])
            else:
                # Perform interpolation
                for name in names:
                    # Interpolation function
                    interp_g = interp1d(self.enei, self.Gsav[name].reshape(len(self.enei), -1), 
                                       axis=0, kind='linear')
                    interp_fr = interp1d(self.enei, self.Frsav[name].reshape(len(self.enei), -1), 
                                        axis=0, kind='linear')
                    interp_fz = interp1d(self.enei, self.Fzsav[name].reshape(len(self.enei), -1), 
                                        axis=0, kind='linear')
                    
                    # Interpolate and reshape
                    self.G[name] = interp_g(enei).reshape(self.Gsav[name].shape[1:])
                    self.Fr[name] = interp_fr(enei).reshape(self.Frsav[name].shape[1:])
                    self.Fz[name] = interp_fz(enei).reshape(self.Fzsav[name].shape[1:])
        
        return self
    
    def set(self, enei, **options):
        """
        Precompute Green function table for given wavelengths.
        
        Parameters:
        -----------
        enei : array_like
            Wavelengths of light in vacuum
        **options : dict
            Options including 'waitbar', 'waitbarlimits'
            
        Returns:
        --------
        self : GreenTabLayer
            Precomputed Green function table
        """
        enei = np.asarray(enei)
        self.enei = enei
        
        # Initialize storage
        first_eval = True
        
        for i, en in enumerate(enei):
            # Evaluate table of Green functions
            self.eval(en, 'new')
            
            if first_eval:
                # Get field names and initialize storage
                names = list(self.G.keys())
                shape = [len(enei)] + list(self.G[names[0]].shape)
                
                self.Gsav = {}
                self.Frsav = {}
                self.Fzsav = {}
                
                for name in names:
                    self.Gsav[name] = np.zeros(shape, dtype=complex)
                    self.Frsav[name] = np.zeros(shape, dtype=complex)
                    self.Fzsav[name] = np.zeros(shape, dtype=complex)
                
                first_eval = False
            
            # Save Green functions
            for name in names:
                self.Gsav[name][i, :] = self.G[name].ravel()
                self.Frsav[name][i, :] = self.Fr[name].ravel()
                self.Fzsav[name][i, :] = self.Fz[name].ravel()
        
        # Reshape arrays
        for name in names:
            self.Gsav[name] = self.Gsav[name].reshape(shape)
            self.Frsav[name] = self.Frsav[name].reshape(shape)
            self.Fzsav[name] = self.Fzsav[name].reshape(shape)
        
        return self
    
    def parset(self, enei, **options):
        """
        Same as set but with parallel computation.
        
        Note: Python implementation uses sequential computation.
        For parallel processing, consider using multiprocessing.
        """
        return self.set(enei, **options)
    
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
        inside : array_like
            True if inside tabulated region
        """
        layer = self.layer
        
        # Round radii and z-values
        r = np.maximum(layer.rmin, r)
        z1, z2 = self._round_layer(layer, z1, z2)
        
        # Inside function
        def is_inside(x, limits):
            return (x >= np.min(limits)) & (x <= np.max(limits))
        
        if len(self.z2) == 1:
            # Uppermost or lowermost layer
            ind1 = self._indlayer(layer, z1)
            ind2 = self._indlayer(layer, z2)
            
            # Find z-values in uppermost or lowermost layer
            in1 = (ind1 == ind2) & (ind1 == 1)
            in2 = (ind1 == ind2) & (ind1 == layer.n + 1)
            inside_layer = in1 | in2
            
            # Determine whether inside
            result = np.zeros_like(inside_layer, dtype=bool)
            if np.any(in1):
                result[in1] = (is_inside(r[in1], self.r) & 
                              is_inside(z1[in1] + self._mindist(layer, z2[in1]), self.z1))
            if np.any(in2):
                result[in2] = (is_inside(r[in2], self.r) & 
                              is_inside(z1[in2] - self._mindist(layer, z2[in2]), self.z1))
            return result
        else:
            return (is_inside(r, self.r) & 
                   is_inside(z1, self.z1) & 
                   is_inside(z2, self.z2))
    
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
            Interpolated Green function components
        r : array_like
            Radius as used in interpolation
        zmin : array_like
            Minimal distance to layers as used in interpolation
        """
        if len(self.z2) == 1:
            return self._interp2d(r, z1, z2)
        else:
            return self._interp3d(r, z1, z2)
    
    def ismember(self, layer, enei=None):
        """
        Determine whether precomputed table is compatible with parameters.
        
        Parameters:
        -----------
        layer : object
            Layer structure
        enei : array_like, optional
            Wavelengths of light in vacuum
            
        Returns:
        --------
        is_compatible : bool
            True if compatible with precomputed Green functions
        """
        # Check if enei is set and compatible
        if (len(self.enei) == 0 or 
            (enei is not None and 
             (np.min(enei) < np.min(self.enei) or np.max(enei) > np.max(self.enei)))):
            return False
        
        # Check layer compatibility
        if (layer.n != self.layer.n or 
            not np.allclose(layer.z, self.layer.z)):
            return False
        
        # Check dielectric functions
        try:
            eps1 = np.array([eps(self.enei) for eps in layer.eps])
            eps2 = np.array([eps(self.enei) for eps in self.layer.eps])
            if np.linalg.norm(eps1 - eps2) > 1e-8:
                return False
        except:
            return False
        
        return True
    
    def norm(self):
        """
        Multiply Green function with distance-dependent factors.
        
        Returns:
        --------
        G, Fr, Fz : dict
            Green function multiplied with distance factors
        """
        assert self.G is not None, "Green function must be evaluated first"
        
        # Tabulated radii and minimal distances
        r = self.pos['r']
        zmin = self.pos['zmin']
        
        # Distance
        d = np.sqrt(r**2 + zmin**2)
        
        # Get field names
        names = list(self.G.keys())
        
        G, Fr, Fz = {}, {}, {}
        for name in names:
            # Green function
            G[name] = self.G[name] * d
            # Radial and z derivatives
            Fr[name] = self.Fr[name] * d**3 / r
            Fz[name] = self.Fz[name] * d**3 / zmin
        
        return G, Fr, Fz
    
    def _interp2d(self, r, z1, z2):
        """Interpolate 2D Green functions in upper/lower medium."""
        # Round radii and z-values
        r = np.maximum(self.layer.rmin, r)
        z1, z2 = self._round_layer(self.layer, z1, z2)
        
        # Combined z-value in uppermost or lowermost layer
        if self._indlayer(self.layer, self.z2[0]) == 1:
            z = z1 + self._mindist(self.layer, self._round_layer(self.layer, z2))
        else:
            z = z1 - self._mindist(self.layer, self._round_layer(self.layer, z2))
        
        # Minimum distance to layer
        zmin = self._mindist(self.layer, z)
        
        # Distance
        d = np.sqrt(r**2 + zmin**2)
        
        # Bilinear interpolation (simplified)
        g, fr, fz = self.norm()
        
        # Get field names
        names = list(self.G.keys())
        
        G, Fr, Fz = {}, {}, {}
        for name in names:
            # Interpolate (simplified implementation)
            G[name] = self._interpolate_2d(g[name], self.r, self.z1, r, z) / d
            Fr[name] = self._interpolate_2d(fr[name], self.r, self.z1, r, z) * r / d**3
            Fz[name] = self._interpolate_2d(fz[name], self.r, self.z1, r, z) * zmin / d**3
        
        return G, Fr, Fz, r, zmin
    
    def _interp3d(self, r, z1, z2):
        """Interpolate 3D Green functions."""
        # Round radii and z-values
        r = np.maximum(self.layer.rmin, r)
        z1, z2 = self._round_layer(self.layer, z1, z2)
        
        # Minimum distance to layer
        zmin = self._mindist(self.layer, z1) + self._mindist(self.layer, z2)
        
        # Distance
        d = np.sqrt(r**2 + zmin**2)
        
        # Trilinear interpolation (simplified)
        g, fr, fz = self.norm()
        
        # Get field names
        names = list(self.G.keys())
        
        G, Fr, Fz = {}, {}, {}
        for name in names:
            # Interpolate (simplified implementation)
            G[name] = self._interpolate_3d(g[name], self.r, self.z1, self.z2, r, z1, z2) / d
            Fr[name] = self._interpolate_3d(fr[name], self.r, self.z1, self.z2, r, z1, z2) * r / d**3
            Fz[name] = self._interpolate_3d(fz[name], self.r, self.z1, self.z2, r, z1, z2) * zmin / d**3
        
        return G, Fr, Fz, r, zmin
    
    # Helper methods (placeholders for actual layer structure operations)
    def _round_layer(self, layer, z1, z2=None):
        """Round z-coordinates to layer interfaces."""
        if z2 is None:
            return z1  # Simplified
        return z1, z2
    
    def _indlayer(self, layer, z):
        """Get layer indices for z-coordinates."""
        return np.ones(np.asarray(z).shape, dtype=int)  # Simplified
    
    def _mindist(self, layer, z):
        """Minimum distance to layer interfaces."""
        return np.abs(np.asarray(z))  # Simplified
    
    def _compute_green_functions(self, layer, enei, r, z1, z2):
        """Compute Green functions (placeholder)."""
        # This would call the actual Green function computation
        shape = (len(r), len(z1), len(z2))
        G = {'p': np.ones(shape, dtype=complex)}
        Fr = {'p': np.ones(shape, dtype=complex)}
        Fz = {'p': np.ones(shape, dtype=complex)}
        pos = {'r': np.ones(shape), 'zmin': np.ones(shape)}
        return G, Fr, Fz, pos
    
    def _interpolate_2d(self, data, x, y, xi, yi):
        """2D interpolation (simplified)."""
        return np.ones_like(xi, dtype=complex)  # Placeholder
    
    def _interpolate_3d(self, data, x, y, z, xi, yi, zi):
        """3D interpolation (simplified)."""
        return np.ones_like(xi, dtype=complex)  # Placeholder


# Alias for backward compatibility
greentablayer = GreenTabLayer