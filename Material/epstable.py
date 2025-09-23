import numpy as np
from scipy import interpolate
import os
from pathlib import Path


class EpsTable:
    """
    Interpolate from tabulated values of dielectric function.
    
    Reads experimental data files containing wavelength-dependent refractive
    index values and provides interpolation for arbitrary wavelengths.
    """
    
    def __init__(self, filename):
        """
        Constructor for tabulated dielectric function.
        
        Parameters:
        -----------
        filename : str
            ASCII file with "ene n k" in each line
            ene : photon energy (eV)
            n   : refractive index (real part)  
            k   : refractive index (imaginary part)
            
        Available files:
            gold.dat, silver.dat            : Johnson, Christy
            goldpalik.dat, silverpalik.dat,
            copperpalik.dat                 : Palik
            
        Usage:
            eps = EpsTable('gold.dat')
        """
        # Load data from file
        ene, n, k = self._load_data_file(filename)
        
        # Conversion factor from eV to nm
        eV2nm = 1239.84193  # hc in eV⋅nm
        
        # Convert energies from eV to nm
        self.enei = eV2nm / ene
        
        # Create splines for interpolation
        # Sort by wavelength for proper interpolation
        sort_idx = np.argsort(self.enei)
        self.enei = self.enei[sort_idx]
        n = n[sort_idx]
        k = k[sort_idx]
        
        # Create interpolation functions
        self.ni = interpolate.interp1d(self.enei, n, kind='cubic', 
                                      bounds_error=True, fill_value=np.nan)
        self.ki = interpolate.interp1d(self.enei, k, kind='cubic',
                                      bounds_error=True, fill_value=np.nan)
        
        # Store wavelength range for validation
        self.enei_min = np.min(self.enei)
        self.enei_max = np.max(self.enei)
    
    def _load_data_file(self, filename):
        """Load data from file, searching in common locations."""
        # Try different possible locations for the data file
        search_paths = [
            filename,  # Current directory
            Path(__file__).parent / 'data' / filename,  # Package data directory
            Path(__file__).parent / filename,  # Same directory as this file
        ]
        
        for filepath in search_paths:
            if Path(filepath).exists():
                return self._read_data_file(filepath)
        
        raise FileNotFoundError(f"Could not find data file '{filename}' in any of: {search_paths}")
    
    def _read_data_file(self, filepath):
        """Read the actual data file."""
        try:
            # Read data, skipping comment lines (starting with %)
            data = np.loadtxt(filepath, comments='%')
            
            if data.shape[1] != 3:
                raise ValueError(f"Data file must have 3 columns (energy, n, k), got {data.shape[1]}")
            
            ene = data[:, 0]  # Energy in eV
            n = data[:, 1]    # Real part of refractive index
            k = data[:, 2]    # Imaginary part of refractive index
            
            return ene, n, k
            
        except Exception as e:
            raise ValueError(f"Error reading data file '{filepath}': {e}")
    
    def __str__(self):
        """String representation."""
        return f"EpsTable: wavelength range {self.enei_min:.1f}-{self.enei_max:.1f} nm"
    
    def __repr__(self):
        """Detailed representation."""
        return f"EpsTable(range={self.enei_min:.1f}-{self.enei_max:.1f} nm, points={len(self.enei)})"
    
    def __call__(self, enei):
        """
        Interpolate tabulated dielectric function and wavenumber.
        
        Parameters:
        -----------
        enei : array_like
            Light wavelength in vacuum (nm)
            
        Returns:
        --------
        eps : array_like
            Interpolated dielectric function
        k : array_like
            Wavenumber in medium
        """
        enei = np.asarray(enei)
        
        # Check wavelength range
        if np.any(enei < self.enei_min) or np.any(enei > self.enei_max):
            raise ValueError(f"Wavelength out of range. Valid range: "
                           f"{self.enei_min:.1f}-{self.enei_max:.1f} nm, "
                           f"requested: {np.min(enei):.1f}-{np.max(enei):.1f} nm")
        
        # Interpolate real and imaginary parts of refractive index
        ni = self.ni(enei)
        ki = self.ki(enei)
        
        # Calculate dielectric function: eps = (n + ik)^2
        n_complex = ni + 1j * ki
        eps = n_complex**2
        
        # Calculate wavenumber
        k = 2 * np.pi / enei * np.sqrt(eps)
        
        return eps, k
    
    def eps_values(self, enei):
        """Get only dielectric function values."""
        eps, _ = self(enei)
        return eps
    
    def wavenumber(self, enei):
        """Get only wavenumber values."""
        _, k = self(enei)
        return k
    
    def refractive_index(self, enei):
        """
        Get complex refractive index n + ik.
        
        Parameters:
        -----------
        enei : array_like
            Wavelengths (nm)
            
        Returns:
        --------
        n_complex : array_like
            Complex refractive index
        """
        enei = np.asarray(enei)
        
        if np.any(enei < self.enei_min) or np.any(enei > self.enei_max):
            raise ValueError(f"Wavelength out of range: {self.enei_min:.1f}-{self.enei_max:.1f} nm")
        
        ni = self.ni(enei)
        ki = self.ki(enei)
        
        return ni + 1j * ki
    
    def plot_data(self, wavelength_range=None):
        """
        Plot the tabulated dielectric function data.
        
        Parameters:
        -----------
        wavelength_range : tuple, optional
            (min_wavelength, max_wavelength) for plotting range
        """
        import matplotlib.pyplot as plt
        
        if wavelength_range is None:
            wl_min, wl_max = self.enei_min, self.enei_max
        else:
            wl_min, wl_max = wavelength_range
            wl_min = max(wl_min, self.enei_min)
            wl_max = min(wl_max, self.enei_max)
        
        wavelengths = np.linspace(wl_min, wl_max, 200)
        eps, _ = self(wavelengths)
        n_complex = self.refractive_index(wavelengths)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Refractive index
        ax1.plot(wavelengths, n_complex.real, 'b-', linewidth=2, label='Real part')
        ax1.set_ylabel('n', fontsize=12)
        ax1.set_title('Refractive Index (Real)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(wavelengths, n_complex.imag, 'r-', linewidth=2, label='Imaginary part')
        ax2.set_ylabel('k', fontsize=12)
        ax2.set_title('Refractive Index (Imaginary)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Dielectric function
        ax3.plot(wavelengths, eps.real, 'b-', linewidth=2)
        ax3.set_xlabel('Wavelength (nm)', fontsize=12)
        ax3.set_ylabel('Re(ε)', fontsize=12)
        ax3.set_title('Dielectric Function (Real)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        ax4.plot(wavelengths, eps.imag, 'r-', linewidth=2)
        ax4.set_xlabel('Wavelength (nm)', fontsize=12)
        ax4.set_ylabel('Im(ε)', fontsize=12)
        ax4.set_title('Dielectric Function (Imaginary)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @property
    def wavelength_range(self):
        """Get the valid wavelength range."""
        return (self.enei_min, self.enei_max)


# Alias for backward compatibility
epstable = EpsTable