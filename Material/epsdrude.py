import numpy as np


class EpsDrude:
    """
    Drude dielectric function.
    
    eps = eps0 - wp^2 / (w * (w + 1i * gammad))
    
    Drude parameters are available for metals Au, Ag, Al.
    """
    
    def __init__(self, name=None):
        """
        Constructor for Drude dielectric function.
        
        Parameters:
        -----------
        name : str, optional
            Material name: 'Au', 'Ag', 'Al' (or 'gold', 'silver', 'aluminum')
            
        Usage:
            eps = EpsDrude('Au')
            eps = EpsDrude('gold')
            eps = EpsDrude()  # Empty constructor
        """
        self.name = name
        self.eps0 = 1      # background dielectric constant
        self.wp = None     # plasmon energy (eV)
        self.gammad = None # damping rate of plasmon (eV)
        
        if name is not None:
            self._init_material(name)
    
    def _init_material(self, name):
        """Initialize Drude dielectric function parameters."""
        # Physical constants (atomic units)
        hartree = 27.2116  # 2 * Rydberg in eV
        tunit = 0.66 / hartree  # time unit in fs
        
        name_lower = name.lower()
        
        if name_lower in ['au', 'gold']:
            rs = 3  # electron gas parameter
            self.eps0 = 10  # background dielectric constant
            gammad = tunit / 10  # Drude relaxation rate
        elif name_lower in ['ag', 'silver']:
            rs = 3
            self.eps0 = 3.3
            gammad = tunit / 30
        elif name_lower in ['al', 'aluminum']:
            rs = 2.07
            self.eps0 = 1
            gammad = 1.06 / hartree
        else:
            raise ValueError(f"Material name '{name}' unknown. Available: Au, Ag, Al")
        
        # Density in atomic units
        density = 3 / (4 * np.pi * rs**3)
        
        # Plasmon energy
        wp = np.sqrt(4 * np.pi * density)
        
        # Save values (convert to eV)
        self.gammad = gammad * hartree
        self.wp = wp * hartree
    
    def __str__(self):
        """String representation."""
        return f"EpsDrude: {self.name}"
    
    def __repr__(self):
        """Detailed string representation."""
        return f"EpsDrude(name='{self.name}', eps0={self.eps0}, wp={self.wp}, gammad={self.gammad})"
    
    def __call__(self, enei):
        """
        Calculate Drude dielectric function and wavenumber.
        
        Parameters:
        -----------
        enei : array_like
            Light wavelength in vacuum (nm)
            
        Returns:
        --------
        eps : array_like
            Drude dielectric function
        k : array_like
            Wavenumber in medium
        """
        if self.wp is None or self.gammad is None:
            raise ValueError("Material not initialized. Use EpsDrude(material_name)")
        
        # Conversion factor from nm to eV
        eV2nm = 1239.84193  # hc in eV⋅nm
        
        enei = np.asarray(enei)
        
        # Convert wavelength (nm) to energy (eV)
        w = eV2nm / enei
        
        # Drude dielectric function
        eps = self.eps0 - self.wp**2 / (w * (w + 1j * self.gammad))
        
        # Wavenumber
        k = 2 * np.pi / enei * np.sqrt(eps)
        
        return eps, k
    
    def eps_values(self, enei):
        """
        Get only the dielectric function values.
        
        Parameters:
        -----------
        enei : array_like
            Wavelengths (nm)
            
        Returns:
        --------
        eps : array_like
            Dielectric function values
        """
        eps, _ = self(enei)
        return eps
    
    def wavenumber(self, enei):
        """
        Get only the wavenumber values.
        
        Parameters:
        -----------
        enei : array_like
            Wavelengths (nm)
            
        Returns:
        --------
        k : array_like
            Wavenumber values
        """
        _, k = self(enei)
        return k
    
    @classmethod
    def available_materials(cls):
        """Return list of available materials."""
        return ['Au', 'gold', 'Ag', 'silver', 'Al', 'aluminum']
    
    def plot_dielectric_function(self, wavelength_range=(300, 800), num_points=100):
        """
        Plot the real and imaginary parts of the dielectric function.
        
        Parameters:
        -----------
        wavelength_range : tuple
            (min_wavelength, max_wavelength) in nm
        num_points : int
            Number of points for plotting
        """
        import matplotlib.pyplot as plt
        
        if self.name is None:
            raise ValueError("Cannot plot: material not specified")
        
        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], num_points)
        eps, _ = self(wavelengths)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        
        # Real part
        ax1.plot(wavelengths, eps.real, 'b-', linewidth=2)
        ax1.set_ylabel('Re(ε)', fontsize=12)
        ax1.set_title(f'Drude Dielectric Function - {self.name}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Imaginary part
        ax2.plot(wavelengths, eps.imag, 'r-', linewidth=2)
        ax2.set_xlabel('Wavelength (nm)', fontsize=12)
        ax2.set_ylabel('Im(ε)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Alias for backward compatibility
epsdrude = EpsDrude