import numpy as np
from scipy.spatial.distance import cdist


class CompGreenStatLayer:
    """
    Green function for layer structure in quasistatic limit.
    
    Works only for single layer and particle located in upper medium.
    The reflected part of the Green functions is computed using image charges.
    """
    
    # Class constants
    name = 'greenfunction'
    needs = [{'sim': 'stat'}, 'layer']
    
    def __init__(self, p1, p2, **options):
        """
        Initialize Green function object for layer structure.
        
        Parameters:
        -----------
        p1 : object
            Green function between points p1 and comparticle p2
        p2 : object
            Green function between points p1 and comparticle p2
        **options : dict
            Options including layer structure
        """
        self.p1 = p1  # Green function between points p1 and comparticle p2
        self.p2 = p2  # Green function between points p1 and comparticle p2
        self.layer = None  # layer structure
        self.g = None      # Green function object (direct term)
        self.gr = None     # Green function object (reflected term)
        
        # Additional properties
        self.p2r = None    # image particle (for reflected Green function)
        self.ind1 = np.array([])  # index to elements of P1 located above layer
        self.ind2 = np.array([])  # index to elements of P2 not located in layer
        self.indl = np.array([])  # index to elements of P2 located in layer
        
        self._init(**options)
    
    def _init(self, **options):
        """Initialize Green function object for layer structure."""
        # Extract layer structure from options
        self.layer = options.get('layer')
        if self.layer is None:
            raise ValueError("Layer structure must be provided in options")
        
        # Make sure that only single layer (substrate)
        assert self.layer.n == 1, "Only single layer supported"
        
        # Index to boundary elements of P2 located in layer
        ind, in_layer = self._indlayer(self.layer, self.p2.pos[:, 2])
        self.indl = np.where(in_layer)[0]
        
        # Make sure all positions of P2 are in upper medium
        assert np.all(ind == self.layer.ind[0]), "All P2 positions must be in upper medium"
        
        # Elements of P2 not located in layer
        self.ind2 = np.setdiff1d(np.arange(self.p2.n), self.indl)
        
        # Elements of P1 located above layer
        ind1, _ = self._indlayer(self.layer, self.p1.pos[:, 2])
        self.ind1 = np.where(ind1 == self.layer.ind[0])[0]
        
        # Vector for displacement (reflection across layer)
        vec = np.array([0, 0, -self.layer.z])
        
        # Create reflected particle
        if len(self.ind2) > 0:
            p2_shifted = self._shift_particle(self.p2, vec)
            p2_selected = self._select_particle(p2_shifted, self.ind2)
            p2_flipped = self._flip_particle(p2_selected, axis=2)  # Flip z-direction
            self.p2r = self._shift_particle(p2_flipped, -vec)
        
        # Create Green function objects
        self.g = self._create_compgreenstat(self.p1, self.p2, **options)
        
        if len(self.ind1) > 0 and len(self.ind2) > 0:
            p1_selected = self._select_particle(self.p1, self.ind1)
            waitbar_options = options.copy()
            waitbar_options['waitbar'] = False
            self.gr = self._create_compgreenstat(p1_selected, self.p2r, **waitbar_options)
    
    def __str__(self):
        """Command window display."""
        return (f"compgreenstatlayer: p1={self.p1}, p2={self.p2}, "
                f"g={self.g}, gr={self.gr}, layer={self.layer}")
    
    def __getattr__(self, name):
        """Handle attribute access for Green function components."""
        if name in ['G', 'F', 'H1', 'H2', 'Gp']:
            return lambda enei: self.eval(enei, name)
        elif name == 'field':
            return self.field
        elif name == 'potential':
            return self.potential
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def eval(self, enei, *keys):
        """
        Evaluate Green function for layer structure.
        
        Parameters:
        -----------
        enei : float
            Light wavelength in vacuum
        *keys : str
            Green function keys ('G', 'F', 'H1', 'H2', 'Gp')
            
        Returns:
        --------
        varargout : list
            Requested Green functions
        """
        # Dielectric functions of medium in upper and lower layer
        eps1 = self.layer.eps[0](enei)  # Upper medium
        eps2 = self.layer.eps[1](enei)  # Lower medium
        
        # Multiplication factors
        f1 = np.full(self.p1.n, 2 * eps1 / (eps2 + eps1))
        f1[~np.isin(np.arange(self.p1.n), self.ind1)] = 1.0  # Only apply to upper medium
        
        fl = 2 * eps1 / (eps1 + eps2)  # Factor for elements in layer
        f2 = -(eps2 - eps1) / (eps2 + eps1)  # Image charge factor
        
        results = []
        
        for key in keys:
            if key == 'G':
                # Direct part
                G = self._get_green_function(self.g, 'G') * f1[:, np.newaxis]
                
                # Correct for contributions in layer
                if len(self.indl) > 0:
                    G[:, self.indl] *= fl
                
                # Add reflected part
                if self.gr is not None and len(self.ind1) > 0 and len(self.ind2) > 0:
                    G_refl = self._get_green_function(self.gr, 'G')
                    G[np.ix_(self.ind1, self.ind2)] += f2 * G_refl
                
                results.append(G)
                
            elif key in ['F', 'H1', 'H2']:
                # Direct part
                F = self._get_green_function(self.g, 'F') * f1[:, np.newaxis]
                
                # Correct for contributions in layer
                ind_above_not_in_layer = np.setdiff1d(self.ind1, self.indl)
                if len(ind_above_not_in_layer) > 0 and len(self.indl) > 0:
                    F[np.ix_(ind_above_not_in_layer, self.indl)] *= (1 + f2)
                
                # Add reflected part
                if self.gr is not None and len(self.ind1) > 0 and len(self.ind2) > 0:
                    F_refl = self._get_green_function(self.gr, 'F')
                    F[np.ix_(self.ind1, self.ind2)] += f2 * F_refl
                
                # Handle diagonal corrections for same particle
                if self.p1 == self.p2:
                    # Set F for flat boundary elements in substrate interface to zero
                    F[np.ix_(self.indl, self.indl)] = 0
                    
                    # Diagonal part correction
                    f_diag = np.zeros(self.p1.n)
                    f_diag[self.indl] = f2
                    
                    if key == 'H1':
                        H1 = F + 2 * np.pi * (np.diag(f_diag) + np.eye(self.p1.n))
                        results.append(H1)
                    elif key == 'H2':
                        H2 = F + 2 * np.pi * (np.diag(f_diag) - np.eye(self.p1.n))
                        results.append(H2)
                    else:
                        results.append(F)
                else:
                    results.append(F)
                    
            elif key == 'Gp':
                # Direct part
                Gp = self._get_green_function(self.g, 'Gp')
                Gp = Gp * f1[np.newaxis, np.newaxis, :]
                
                # Correct for contributions in layer
                if len(self.indl) > 0:
                    Gp[:, :, self.indl] *= fl
                
                # Add reflected part
                if self.gr is not None and len(self.ind1) > 0 and len(self.ind2) > 0:
                    Gp_refl = self._get_green_function(self.gr, 'Gp')
                    Gp[self.ind1, :, :][:, :, self.ind2] += f2 * Gp_refl
                
                results.append(Gp)
                
            elif key == 'd':
                # Distance matrix
                d = cdist(self.p1.pos, self.p2.pos)
                results.append(d)
        
        return results[0] if len(results) == 1 else results
    
    def field(self, sig, inout=1):
        """
        Electric field inside/outside of particle surface.
        
        Parameters:
        -----------
        sig : object
            compstruct with surface charges
        inout : int, optional
            Fields inside (1, default) or outside (2) of particle
            
        Returns:
        --------
        field : object
            compstruct object with electric field
        """
        # Get distance and derivative of Green function
        d, Gp = self.eval(sig.enei, 'd', 'Gp')
        
        # Electric field calculation
        if inout == 1:
            # Inside: E = -(Gp + 2π * nvec ⊗ δ) * σ
            correction = 2 * np.pi * self._outer_product(
                getattr(self.p1, 'nvec', np.eye(3)[:self.p1.n, :]),
                (d == 0)
            )
            e = -self._matmul(Gp + correction, getattr(sig, 'sig', np.ones(self.p1.n)))
        else:
            # Outside: E = -(Gp - 2π * nvec ⊗ δ) * σ
            correction = 2 * np.pi * self._outer_product(
                getattr(self.p1, 'nvec', np.eye(3)[:self.p1.n, :]),
                (d == 0)
            )
            e = -self._matmul(Gp - correction, getattr(sig, 'sig', np.ones(self.p1.n)))
        
        # Create output structure
        return self._create_compstruct(self.p1, sig.enei, e=e)
    
    def potential(self, sig, inout=1):
        """
        Determine potentials and surface derivatives inside/outside of particle.
        
        Parameters:
        -----------
        sig : object
            compstruct with surface charges
        inout : int, optional
            Potentials inside (1, default) or outside (2) of particle
            
        Returns:
        --------
        pot : object
            compstruct object with potentials & surface derivatives
        """
        # Get Green function and surface derivative
        H_key = 'H1' if inout == 1 else 'H2'
        G, H = self.eval(sig.enei, 'G', H_key)
        
        # Calculate potential and surface derivative
        sigma = getattr(sig, 'sig', np.ones(self.p2.n))
        phi = self._matmul(G, sigma)
        phip = self._matmul(H, sigma)
        
        # Create output structure
        if inout == 1:
            return self._create_compstruct(self.p1, sig.enei, phi1=phi, phi1p=phip)
        else:
            return self._create_compstruct(self.p1, sig.enei, phi2=phi, phi2p=phip)
    
    # Helper methods
    def _indlayer(self, layer, z_coords):
        """Determine layer indices for z-coordinates."""
        # Simplified implementation
        ind = np.ones(len(z_coords), dtype=int) * layer.ind[0]
        in_layer = np.abs(z_coords - layer.z) < 1e-10
        return ind, in_layer
    
    def _shift_particle(self, particle, vec):
        """Shift particle by vector."""
        # Placeholder - would modify particle positions
        class ShiftedParticle:
            def __init__(self, original, vec):
                self.original = original
                self.vec = vec
                self.pos = original.pos + vec
                self.n = original.n
        return ShiftedParticle(particle, vec)
    
    def _select_particle(self, particle, indices):
        """Select subset of particle elements."""
        class SelectedParticle:
            def __init__(self, original, indices):
                self.original = original
                self.indices = indices
                self.pos = original.pos[indices, :]
                self.n = len(indices)
        return SelectedParticle(particle, indices)
    
    def _flip_particle(self, particle, axis):
        """Flip particle along specified axis."""
        class FlippedParticle:
            def __init__(self, original, axis):
                self.original = original
                self.axis = axis
                self.pos = original.pos.copy()
                self.pos[:, axis] *= -1
                self.n = original.n
        return FlippedParticle(particle, axis)
    
    def _create_compgreenstat(self, p1, p2, **options):
        """Create a compgreenstat object (placeholder)."""
        class PlaceholderCompGreenStat:
            def __init__(self, p1, p2, **options):
                self.p1 = p1
                self.p2 = p2
                
            def eval(self, key):
                n1, n2 = self.p1.n, self.p2.n
                if key in ['G', 'F']:
                    return np.eye(n1, n2, dtype=complex)
                elif key == 'Gp':
                    return np.zeros((n1, 3, n2), dtype=complex)
                return None
                
        return PlaceholderCompGreenStat(p1, p2, **options)
    
    def _get_green_function(self, green_obj, key):
        """Get Green function from object."""
        if hasattr(green_obj, 'eval'):
            return green_obj.eval(key)
        return np.eye(self.p1.n, self.p2.n, dtype=complex)
    
    def _create_compstruct(self, p, enei, **kwargs):
        """Create a compstruct object."""
        class PlaceholderCompStruct:
            def __init__(self, p, enei, **kwargs):
                self.p = p
                self.enei = enei
                for key, value in kwargs.items():
                    setattr(self, key, value)
        return PlaceholderCompStruct(p, enei, **kwargs)
    
    def _matmul(self, A, x):
        """Matrix multiplication."""
        if A.ndim == 3:  # Gp case
            return np.tensordot(A, x, axes=([2], [0]))
        return np.dot(A, x)
    
    def _outer_product(self, nvec, mask):
        """Outer product for correction terms."""
        return np.outer(nvec.ravel(), mask.ravel()).reshape(nvec.shape[0], mask.shape[0])


# Alias for backward compatibility
compgreenstatlayer = CompGreenStatLayer