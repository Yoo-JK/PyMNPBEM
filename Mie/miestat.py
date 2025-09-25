"""
PyMNPBEM - Mie theory for spherical particles using quasistatic approximation
Converted from MATLAB MNPBEM miestat module
"""

import numpy as np
from scipy.special import kn  # Modified Bessel function of the second kind
from typing import Tuple, Optional, Union
import warnings
from ..core.bembase import BEMBase
from ..utils.sphtable import sphtable
from ..utils.atomic_units import AtomicUnits


class MieStat(BEMBase):
    """
    Mie theory for spherical particle using the quasistatic approximation.
    Programs are only intended for testing.
    """
    
    name = 'miesolver'
    needs = {'sim': 'stat'}
    
    def __init__(self, epsin, epsout, diameter, lmax=None, **kwargs):
        """
        Initialize spherical particle for quasistatic Mie theory.
        
        Parameters:
        -----------
        epsin : callable
            Dielectric function inside of sphere
        epsout : callable  
            Dielectric function outside of sphere
        diameter : float
            Sphere diameter
        lmax : int, optional
            Maximum number for spherical harmonic degrees
        """
        super().__init__(**kwargs)
        
        self.epsin = epsin
        self.epsout = epsout
        self.diameter = diameter
        
        # Set default lmax if not provided
        if lmax is None:
            lmax = 10
            
        # Generate spherical harmonics tables
        self.ltab, self.mtab = sphtable(lmax, mode='full')
    
    def __str__(self):
        """String representation."""
        return f"MieStat(epsin={self.epsin}, epsout={self.epsout}, diameter={self.diameter})"
    
    def __repr__(self):
        return self.__str__()
    
    def absorption(self, enei: Union[float, np.ndarray]) -> np.ndarray:
        """
        Absorption cross section for miestat objects.
        
        Parameters:
        -----------
        enei : float or array_like
            Wavelength of light in vacuum
            
        Returns:
        --------
        abs : ndarray
            Absorption cross section
        """
        enei = np.asarray(enei)
        
        # Background dielectric function
        epsb = self.epsout(0)
        
        # Refractive index
        nb = np.sqrt(epsb)
        
        # Wavevector of light
        k = 2 * np.pi / enei * nb
        
        # Ratio of dielectric functions
        epsz = self.epsin(enei) / epsb
        
        # Sphere radius
        a = self.diameter / 2
        
        # Polarizability of sphere (see van de Hulst, Sec. 6.31)
        alpha = (epsz - 1) / (epsz + 2) * a**3
        
        # Absorption cross section
        abs_cross_section = 4 * np.pi * k * np.imag(alpha)
        
        return abs_cross_section
    
    def scattering(self, enei: Union[float, np.ndarray]) -> np.ndarray:
        """
        Scattering cross section for miestat objects.
        
        Parameters:
        -----------
        enei : float or array_like
            Wavelength of light in vacuum
            
        Returns:
        --------
        sca : ndarray
            Scattering cross section
        """
        enei = np.asarray(enei)
        
        # Background dielectric function
        epsb = self.epsout(0)
        
        # Refractive index
        nb = np.sqrt(epsb)
        
        # Wavevector of light
        k = 2 * np.pi / enei * nb
        
        # Ratio of dielectric functions
        epsz = self.epsin(enei) / epsb
        
        # Sphere radius
        a = self.diameter / 2
        
        # Polarizability of sphere (see van de Hulst, Sec. 6.31)
        alpha = (epsz - 1) / (epsz + 2) * a**3
        
        # Scattering cross section
        sca = 8 * np.pi / 3 * k**4 * np.abs(alpha)**2
        
        return sca
    
    def extinction(self, enei: Union[float, np.ndarray]) -> np.ndarray:
        """
        Extinction cross section for miestat objects.
        
        Parameters:
        -----------
        enei : float or array_like
            Wavelength of light in vacuum
            
        Returns:
        --------
        ext : ndarray
            Extinction cross section
        """
        return self.scattering(enei) + self.absorption(enei)
    
    def decayrate(self, enei: float, z: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Total and radiative decay rate for oscillating dipole.
        Scattering rates are given in units of the free-space decay rate.
        
        Parameters:
        -----------
        enei : float
            Wavelength of light in vacuum (only one value)
        z : float or array_like
            Dipole positions on z axis
            
        Returns:
        --------
        tot : ndarray
            Total scattering rate for dipole orientations x and z
        rad : ndarray
            Radiative scattering rate for dipole orientations x and z
        """
        if not np.isscalar(enei):
            raise ValueError("enei must be a single value")
            
        z = np.asarray(z)
        
        # Background dielectric function
        epsb = self.epsout(0)
        
        # Refractive index
        nb = np.sqrt(epsb)
        
        # Free space radiative lifetime (Wigner-Weisskopf)
        gamma = 4/3 * nb * (2 * np.pi / enei)**3
        
        # Ratio of dielectric functions
        epsz = self.epsin(enei) / epsb
        
        # Total and radiative scattering rate
        tot = np.zeros((len(z), 2))
        rad = np.zeros((len(z), 2))
        
        # Dipole orientations
        dipole_orientations = np.array([[1, 0, 0], [0, 0, 1]])
        
        # Loop over dipole positions and orientation
        for iz, z_pos in enumerate(z):
            for idip in range(2):
                # Dipole orientation
                dip = dipole_orientations[idip]
                
                # Position of dipole
                pos = np.array([0, 0, z_pos]) / self.diameter
                
                # Spherical harmonics coefficients for dipole
                adip = self._adipole(pos, dip)
                
                # Induced dipole moment of sphere
                indip = self._dipole(adip, epsz)
                
                # Radiative decay rate (in units of free decay rate)
                rad[iz, idip] = np.linalg.norm(dip + indip)**2
                
                # Induced electric field
                efield = self._field(pos, adip, epsz) / (epsb * self.diameter**3)
                
                # Total decay rate
                tot[iz, idip] = rad[iz, idip] + np.imag(np.dot(efield, dip)) / (gamma / 2)
        
        return tot, rad
    
    def loss(self, b: Union[float, np.ndarray], enei: Union[float, np.ndarray], 
             beta: float = 0.7) -> np.ndarray:
        """
        Energy loss of fast electron in vicinity of dielectric sphere.
        See F. J. Garcia de Abajo, Rev. Mod. Phys. 82, 209 (2010), Eq. (31).
        
        Parameters:
        -----------
        b : float or array_like
            Impact parameter
        enei : float or array_like
            Wavelength of light in vacuum
        beta : float, optional
            Electron velocity in units of speed of light (default 0.7)
            
        Returns:
        --------
        prob : ndarray
            EELS probability
        """
        b = np.asarray(b)
        enei = np.asarray(enei)
        
        # Make sure that electron trajectory does not penetrate sphere
        if np.any(b <= 0):
            raise ValueError("All impact parameters must be positive")
        
        # Radius of sphere
        a = 0.5 * self.diameter
        
        # Add sphere radius to impact parameter
        b = a + b
        
        # EELS probability
        prob = np.zeros((len(b), len(enei)))
        
        # Spherical harmonics
        l, m = sphtable(np.max(self.ltab), mode='full')
        
        for ien, energy in enumerate(enei):
            # Wavenumber of light in medium
            epsb, k = self.epsout(energy)
            
            # Mie expressions only implemented for epsb = 1
            if epsb != 1:
                warnings.warn("Mie expressions only implemented for epsb = 1")
                continue
            
            # Relative dielectric function
            epsz = self.epsin(energy) / epsb
            
            # Polarizability of sphere
            alpha = (l * epsz - l) / (l * epsz + l + 1) * a**3
            
            # Prefactor
            fac = (k * a / beta)**(2 * l) / (np.math.factorial(l + m) * np.math.factorial(l - m))
            
            for ib, b_val in enumerate(b):
                # Modified Bessel function of the second kind
                K = kn(m, k * b_val / beta)
                
                # Energy loss probability
                prob[ib, ien] = np.sum(fac * K**2 * np.imag(alpha))
        
        # Load atomic units
        au = AtomicUnits()
        
        # Convert to units of (1/eV)
        prob = 4 * au.fine**2 / (np.pi * au.hartree * au.bohr * beta**2 * a**2) * prob
        
        return prob
    
    def _adipole(self, pos: np.ndarray, dip: np.ndarray) -> np.ndarray:
        """
        Calculate spherical harmonics coefficients for dipole.
        This is a placeholder - actual implementation would depend on the specific
        spherical harmonics expansion used in the original code.
        """
        # Placeholder implementation
        # In the actual implementation, this would compute the multipole expansion
        # coefficients for a dipole at position pos with moment dip
        return np.zeros(len(self.ltab))
    
    def _dipole(self, adip: np.ndarray, epsz: complex) -> np.ndarray:
        """
        Calculate induced dipole moment of sphere.
        This is a placeholder - actual implementation would depend on the Mie theory.
        """
        # Placeholder implementation
        return np.zeros(3)
    
    def _field(self, pos: np.ndarray, adip: np.ndarray, epsz: complex) -> np.ndarray:
        """
        Calculate induced electric field.
        This is a placeholder - actual implementation would compute the field
        from the multipole expansion.
        """
        # Placeholder implementation
        return np.zeros(3)


# For backward compatibility with MATLAB-style function calls
def absorption(obj: MieStat, enei: Union[float, np.ndarray]) -> np.ndarray:
    """Wrapper function for absorption cross section."""
    return obj.absorption(enei)


def scattering(obj: MieStat, enei: Union[float, np.ndarray]) -> np.ndarray:
    """Wrapper function for scattering cross section."""
    return obj.scattering(enei)


def extinction(obj: MieStat, enei: Union[float, np.ndarray]) -> np.ndarray:
    """Wrapper function for extinction cross section."""
    return obj.extinction(enei)


def decayrate(obj: MieStat, enei: float, z: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper function for decay rate calculation."""
    return obj.decayrate(enei, z)


def loss(obj: MieStat, b: Union[float, np.ndarray], enei: Union[float, np.ndarray], 
         beta: float = 0.7) -> np.ndarray:
    """Wrapper function for EELS loss probability."""
    return obj.loss(b, enei, beta)