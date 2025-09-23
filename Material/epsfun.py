import numpy as np


class EpsFun:
    """
    Dielectric function using user-supplied function.
    """
    
    def __init__(self, fun, key='nm'):
        """
        Constructor for EpsFun.
        
        Parameters:
        -----------
        fun : callable
            Function for evaluation eps = fun(enei)
        key : str, optional
            'nm' for wavelengths or 'eV' for energies (default: 'nm')
        """
        self.fun = fun
        self.key = key
    
    def __str__(self):
        """Command window display."""
        return f"epsfun: fun={self.fun}, key='{self.key}'"
    
    def __call__(self, enei):
        """
        Evaluate dielectric function.
        
        Parameters:
        -----------
        enei : array_like
            Light wavelength in vacuum (nm)
            
        Returns:
        --------
        eps : array_like
            Dielectric function
        k : array_like
            Wavenumber in medium
        """
        enei = np.asarray(enei)
        
        # Evaluate dielectric function
        if self.key == 'nm':
            eps = self.fun(enei)
        elif self.key == 'eV':
            eV2nm = 1239.84193  # hc in eV⋅nm
            eps = self.fun(eV2nm / enei)
        else:
            raise ValueError("key must be 'nm' or 'eV'")
        
        # Wavenumber
        k = 2 * np.pi / enei * np.sqrt(eps)
        
        return eps, k


# Alias for backward compatibility
epsfun = EpsFun