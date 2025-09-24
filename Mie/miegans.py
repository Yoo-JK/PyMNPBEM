"""
miegans.py - Mie-Gans Theory for Ellipsoidal Particles
Converted from MATLAB @miegans class

This module implements quasistatic Mie-Gans theory for ellipsoidal particles:
- Depolarization factors computation
- Scattering cross section
- Absorption cross section
- Extinction cross section
"""

import numpy as np
from typing import Callable, Union, Optional
from scipy.integrate import quad

class MieGans:
    """
    Mie-Gans theory for ellipsoidal particle and quasistatic approximation.
    
    This class implements the Mie-Gans theory for calculating optical cross sections
    of ellipsoidal particles in the quasistatic approximation.
    """
    
    def __init__(self, epsin: Callable, epsout: Callable, axes: np.ndarray):
        """
        Initialize ellipsoidal particle for quasistatic Mie-Gans theory.
        
        Parameters:
        -----------
        epsin : callable
            Dielectric function inside the ellipsoid
            Function signature: epsin(wavelength) -> complex
        epsout : callable
            Dielectric function outside the ellipsoid
            Function signature: epsout(wavelength) -> complex
        axes : np.ndarray
            Ellipsoid axes [a, b, c]
        """
        self.epsin = epsin
        self.epsout = epsout
        self.ax = np.asarray(axes, dtype=float)
        
        # Private properties for depolarization factors
        self._L1 = None
        self._L2 = None
        self._L3 = None
        
        # Compute depolarization factors
        self._init()
    
    def _init(self):
        """
        Compute depolarization factors for ellipsoid.
        
        Based on H.C. van de Hulst, Light scattering by small particles, Sec. 6.32.
        """
        a, b, c = self.ax
        
        # Define integrand functions for depolarization factors
        def integrand_L1(s):
            return a * b * c / (2 * (s + a**2)**1.5 * (s + b**2)**0.5 * (s + c**2)**0.5)
        
        def integrand_L2(s):
            return a * b * c / (2 * (s + a**2)**0.5 * (s + b**2)**1.5 * (s + c**2)**0.5)
        
        def integrand_L3(s):
            return a * b * c / (2 * (s + a**2)**0.5 * (s + b**2)**0.5 * (s + c**2)**1.5)
        
        # Integration limits
        upper_limit = 1e5 * np.max(self.ax)
        
        # Compute depolarization factors using numerical integration
        try:
            self._L1, _ = quad(integrand_L1, 0, upper_limit, epsabs=1e-8, epsrel=1e-8)
            self._L2, _ = quad(integrand_L2, 0, upper_limit, epsabs=1e-8, epsrel=1e-8)
            self._L3, _ = quad(integrand_L3, 0, upper_limit, epsabs=1e-8, epsrel=1e-8)
        except Exception as e:
            raise RuntimeError(f"Failed to compute depolarization factors: {e}")
    
    @property
    def L1(self):
        """Depolarization factor L1."""
        return self._L1
    
    @property
    def L2(self):
        """Depolarization factor L2."""
        return self._L2
    
    @property
    def L3(self):
        """Depolarization factor L3."""
        return self._L3
    
    def absorption(self, enei: Union[float, np.ndarray], pol: np.ndarray) -> Union[float, np.ndarray]:
        """
        Absorption cross section for ellipsoidal particle.
        
        Parameters:
        -----------
        enei : float or np.ndarray
            Wavelength of light in vacuum
        pol : np.ndarray
            Light polarization vector [pol_x, pol_y, pol_z]
            
        Returns:
        --------
        float or np.ndarray
            Absorption cross section
        """
        # Background dielectric function
        epsb = self.epsout(0)
        
        # Refractive index
        nb = np.sqrt(epsb)
        
        # Wave vector of light
        k = 2 * np.pi / enei * nb
        
        # Ratio of dielectric functions
        epsz = self.epsin(enei) / epsb
        
        # Volume of ellipsoid
        vol = 4 * np.pi / 3 * np.prod(self.ax / 2)
        
        # Polarizabilities
        a1 = vol / (4 * np.pi) / (self._L1 + 1 / (epsz - 1))
        a2 = vol / (4 * np.pi) / (self._L2 + 1 / (epsz - 1))
        a3 = vol / (4 * np.pi) / (self._L3 + 1 / (epsz - 1))
        
        # Absorption cross section
        alpha_total = a1 * pol[0] + a2 * pol[1] + a3 * pol[2]
        abs_cross_section = 4 * np.pi * k * np.imag(alpha_total)
        
        return abs_cross_section
    
    def scattering(self, enei: Union[float, np.ndarray], pol: np.ndarray) -> Union[float, np.ndarray]:
        """
        Scattering cross section for ellipsoidal particle.
        
        Parameters:
        -----------
        enei : float or np.ndarray
            Wavelength of light in vacuum
        pol : np.ndarray
            Light polarization vector [pol_x, pol_y, pol_z]
            
        Returns:
        --------
        float or np.ndarray
            Scattering cross section
        """
        # Background dielectric function
        epsb = self.epsout(0)
        
        # Refractive index
        nb = np.sqrt(epsb)
        
        # Wave vector of light
        k = 2 * np.pi / enei * nb
        
        # Ratio of dielectric functions
        epsz = self.epsin(enei) / epsb
        
        # Volume of ellipsoid
        vol = 4 * np.pi / 3 * np.prod(self.ax / 2)
        
        # Polarizabilities
        a1 = vol / (4 * np.pi) / (self._L1 + 1 / (epsz - 1))
        a2 = vol / (4 * np.pi) / (self._L2 + 1 / (epsz - 1))
        a3 = vol / (4 * np.pi) / (self._L3 + 1 / (epsz - 1))
        
        # Scattering cross section
        sca_cross_section = 8 * np.pi / 3 * k**4 * (
            np.abs(a1 * pol[0])**2 + 
            np.abs(a2 * pol[1])**2 + 
            np.abs(a3 * pol[2])**2
        )
        
        return sca_cross_section
    
    def extinction(self, enei: Union[float, np.ndarray], pol: np.ndarray) -> Union[float, np.ndarray]:
        """
        Extinction cross section for ellipsoidal particle.
        
        Parameters:
        -----------
        enei : float or np.ndarray
            Wavelength of light in vacuum
        pol : np.ndarray
            Light polarization vector [pol_x, pol_y, pol_z]
            
        Returns:
        --------
        float or np.ndarray
            Extinction cross section
        """
        return self.scattering(enei, pol) + self.absorption(enei, pol)
    
    # Convenient aliases
    def sca(self, enei: Union[float, np.ndarray], pol: np.ndarray) -> Union[float, np.ndarray]:
        """Alias for scattering method."""
        return self.scattering(enei, pol)
    
    def abs(self, enei: Union[float, np.ndarray], pol: np.ndarray) -> Union[float, np.ndarray]:
        """Alias for absorption method."""
        return self.absorption(enei, pol)
    
    def ext(self, enei: Union[float, np.ndarray], pol: np.ndarray) -> Union[float, np.ndarray]:
        """Alias for extinction method."""
        return self.extinction(enei, pol)
    
    def __str__(self):
        """String representation of the object."""
        return f"MieGans:\n  axes: {self.ax}\n  L1: {self._L1:.6f}\n  L2: {self._L2:.6f}\n  L3: {self._L3:.6f}"
    
    def __repr__(self):
        """Object representation."""
        return f"MieGans(axes={self.ax})"

# Convenience functions for creating common ellipsoids
def create_sphere(epsin: Callable, epsout: Callable, radius: float) -> MieGans:
    """
    Create a spherical particle (special case of ellipsoid).
    
    Parameters:
    -----------
    epsin : callable
        Dielectric function inside
    epsout : callable  
        Dielectric function outside
    radius : float
        Sphere radius
        
    Returns:
    --------
    MieGans
        MieGans object for spherical particle
    """
    axes = np.array([2*radius, 2*radius, 2*radius])
    return MieGans(epsin, epsout, axes)

def create_spheroid(epsin: Callable, epsout: Callable, 
                   semi_major: float, semi_minor: float, 
                   oblate: bool = True) -> MieGans:
    """
    Create a spheroidal particle.
    
    Parameters:
    -----------
    epsin : callable
        Dielectric function inside
    epsout : callable
        Dielectric function outside
    semi_major : float
        Semi-major axis length
    semi_minor : float
        Semi-minor axis length
    oblate : bool
        If True, create oblate spheroid (flattened)
        If False, create prolate spheroid (elongated)
        
    Returns:
    --------
    MieGans
        MieGans object for spheroidal particle
    """
    if oblate:
        # Oblate spheroid: a = b > c
        axes = np.array([2*semi_major, 2*semi_major, 2*semi_minor])
    else:
        # Prolate spheroid: a > b = c  
        axes = np.array([2*semi_major, 2*semi_minor, 2*semi_minor])
    
    return MieGans(epsin, epsout, axes)

# Example dielectric functions
def drude_metal(wavelength, plasma_freq=1.37e16, gamma=8.5e13):
    """
    Drude model dielectric function for metals.
    
    Parameters:
    -----------
    wavelength : float or np.ndarray
        Wavelength in meters
    plasma_freq : float
        Plasma frequency in rad/s
    gamma : float
        Damping frequency in rad/s
        
    Returns:
    --------
    complex
        Dielectric function value
    """
    # Convert wavelength to frequency
    c = 2.998e8  # speed of light
    omega = 2 * np.pi * c / wavelength
    
    # Drude model
    eps = 1 - plasma_freq**2 / (omega**2 + 1j * gamma * omega)
    return eps

def constant_dielectric(eps_value):
    """
    Create a constant dielectric function.
    
    Parameters:
    -----------
    eps_value : complex
        Constant dielectric function value
        
    Returns:
    --------
    callable
        Dielectric function
    """
    def eps_func(wavelength):
        return eps_value
    return eps_func

# Example usage functions
def compute_spectrum(mie_obj: MieGans, wavelengths: np.ndarray, 
                    polarization: np.ndarray = None) -> dict:
    """
    Compute optical spectrum for MieGans object.
    
    Parameters:
    -----------
    mie_obj : MieGans
        MieGans object
    wavelengths : np.ndarray
        Array of wavelengths
    polarization : np.ndarray, optional
        Polarization vector [px, py, pz]. Default is [1, 0, 0]
        
    Returns:
    --------
    dict
        Dictionary with 'wavelengths', 'absorption', 'scattering', 'extinction'
    """
    if polarization is None:
        polarization = np.array([1.0, 0.0, 0.0])  # x-polarized light
    
    absorption = np.zeros_like(wavelengths, dtype=complex)
    scattering = np.zeros_like(wavelengths, dtype=complex)
    extinction = np.zeros_like(wavelengths, dtype=complex)
    
    for i, wl in enumerate(wavelengths):
        absorption[i] = mie_obj.absorption(wl, polarization)
        scattering[i] = mie_obj.scattering(wl, polarization)
        extinction[i] = mie_obj.extinction(wl, polarization)
    
    return {
        'wavelengths': wavelengths,
        'absorption': absorption,
        'scattering': scattering,
        'extinction': extinction
    }