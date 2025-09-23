import numpy as np


class EpsConst:
    """
    Dielectric constant.
    
    A simple class representing a material with constant dielectric properties
    that do not depend on wavelength.
    """
    
    def __init__(self, eps):
        """
        Set dielectric constant to given value.
        
        Parameters:
        -----------
        eps : float or complex
            Dielectric constant value
            
        Usage:
            eps = EpsConst(1.33**2)  # Water
            eps = EpsConst(1.0)      # Vacuum/air
        """
        self.eps = eps
    
    def __str__(self):
        """String representation for print()."""
        return f"EpsConst: {self.eps}"
    
    def __repr__(self):
        """String representation for interactive display."""
        return f"EpsConst({self.eps})"
    
    def __call__(self, enei):
        """
        Get dielectric constant and wavenumber for given wavelengths.
        
        Parameters:
        -----------
        enei : array_like
            Light wavelength in vacuum
            
        Returns:
        --------
        eps : array_like
            Dielectric constant (constant for all wavelengths)
        k : array_like
            Wavenumber in medium
            
        Usage:
            eps_val, k_val = eps_obj(wavelengths)
        """
        enei = np.asarray(enei)
        eps_array = np.full_like(enei, self.eps, dtype=complex)
        k = self.wavenumber(enei)
        return eps_array, k
    
    def wavenumber(self, enei):
        """
        Calculate wavenumber in medium.
        
        Parameters:
        -----------
        enei : array_like
            Light wavelength in vacuum
            
        Returns:
        --------
        k : array_like
            Wavenumber in medium
        """
        enei = np.asarray(enei)
        k = 2 * np.pi / enei * np.sqrt(self.eps)
        return k
    
    def eps_values(self, enei):
        """
        Get dielectric constant values for given wavelengths.
        
        Parameters:
        -----------
        enei : array_like
            Wavelengths
            
        Returns:
        --------
        eps : array_like
            Dielectric constant values
        """
        enei = np.asarray(enei)
        return np.full_like(enei, self.eps, dtype=complex)


# Alias for backward compatibility and consistency with MATLAB naming
epsconst = EpsConst