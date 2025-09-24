"""
mieret.py - Full Mie Theory for Spherical Particles
Converted from MATLAB @mieret class

This module implements the full Mie theory using Maxwell equations:
- Scattering, absorption, extinction cross sections
- Decay rate calculations for oscillating dipoles
- EELS (Electron Energy Loss Spectroscopy) calculations
- Riccati-Bessel functions and Mie coefficients
"""

import numpy as np
from typing import Callable, Union, Optional, Tuple
from scipy.special import spherical_jn, spherical_yn, jv, yv, kv
from scipy.integrate import quad

class MieRet:
    """
    Mie theory for spherical particle using the full Maxwell equations.
    
    This class implements the complete Mie theory for spherical particles,
    including retardation effects and the full electromagnetic field solution.
    """
    
    name = 'miesolver'
    needs = {'sim': 'ret'}
    
    def __init__(self, epsin: Callable, epsout: Callable, diameter: float, 
                 lmax: int = 20, **kwargs):
        """
        Initialize spherical particle for full Mie theory.
        
        Parameters:
        -----------
        epsin : callable
            Dielectric function inside the sphere
        epsout : callable
            Dielectric function outside the sphere
        diameter : float
            Sphere diameter
        lmax : int, optional
            Maximum spherical harmonic degree (default: 20)
        """
        self.epsin = epsin
        self.epsout = epsout
        self.diameter = diameter
        
        # Initialize spherical harmonic tables
        self.ltab, self.mtab = self._create_sph_table(lmax)
    
    def _create_sph_table(self, lmax: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create spherical harmonics table."""
        ltab = []
        mtab = []
        
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                ltab.append(l)
                mtab.append(m)
        
        return np.array(ltab), np.array(mtab)
    
    def _riccati_bessel(self, z: Union[float, np.ndarray], 
                       ltab: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Riccati-Bessel functions.
        
        Based on Abramowitz and Stegun, Handbook of Mathematical Functions, Chap. 10.
        
        Parameters:
        -----------
        z : float or np.ndarray
            Argument
        ltab : np.ndarray
            Angular momentum components
            
        Returns:
        --------
        j, h, zjp, zhp : np.ndarray
            Spherical Bessel functions and their derivatives
        """
        z = np.asarray(z)
        lmax = np.max(ltab)
        l_range = np.arange(1, lmax + 1)
        
        # Spherical Bessel functions of first and second kind
        if z.ndim == 0:  # scalar case
            j0 = np.sin(z) / z if z != 0 else 1.0
            y0 = -np.cos(z) / z if z != 0 else -np.inf
            
            j = np.sqrt(np.pi / (2 * z)) * jv(l_range + 0.5, z) if z != 0 else np.zeros(len(l_range))
            y = np.sqrt(np.pi / (2 * z)) * yv(l_range + 0.5, z) if z != 0 else np.full(len(l_range), -np.inf)
        else:  # array case
            j0 = np.where(z != 0, np.sin(z) / z, 1.0)
            y0 = np.where(z != 0, -np.cos(z) / z, -np.inf)
            
            j = np.zeros((len(z), len(l_range)))
            y = np.zeros((len(z), len(l_range)))
            
            for i, zi in enumerate(z):
                if zi != 0:
                    j[i] = np.sqrt(np.pi / (2 * zi)) * jv(l_range + 0.5, zi)
                    y[i] = np.sqrt(np.pi / (2 * zi)) * yv(l_range + 0.5, zi)
                else:
                    j[i] = 0
                    y[i] = -np.inf
        
        # Spherical Bessel function of third kind
        h0 = j0 + 1j * y0
        h = j + 1j * y
        
        # Derivatives
        if z.ndim == 0:
            j_prev = np.concatenate([[j0], j[:-1]]) if len(j) > 0 else np.array([j0])
            h_prev = np.concatenate([[h0], h[:-1]]) if len(h) > 0 else np.array([h0])
            
            zjp = z * j_prev - l_range * j
            zhp = z * h_prev - l_range * h
        else:
            zjp = np.zeros_like(j)
            zhp = np.zeros_like(h)
            
            for i, zi in enumerate(z):
                if len(j.shape) > 1:
                    j_prev = np.concatenate([[j0[i]], j[i, :-1]]) if j.shape[1] > 0 else np.array([j0[i]])
                    h_prev = np.concatenate([[h0[i]], h[i, :-1]]) if h.shape[1] > 0 else np.array([h0[i]])
                    
                    zjp[i] = zi * j_prev - l_range * j[i]
                    zhp[i] = zi * h_prev - l_range * h[i]
        
        # Select values for requested ltab
        if z.ndim == 0:
            j_out = j[ltab - 1]
            h_out = h[ltab - 1]
            zjp_out = zjp[ltab - 1]
            zhp_out = zhp[ltab - 1]
        else:
            j_out = j[:, ltab - 1]
            h_out = h[:, ltab - 1]
            zjp_out = zjp[:, ltab - 1]
            zhp_out = zhp[:, ltab - 1]
        
        return j_out, h_out, zjp_out, zhp_out
    
    def _mie_coefficients(self, k: float, diameter: float, epsr: complex, 
                         mur: float, l: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Mie coefficients according to Bohren and Huffman (1983).
        
        Parameters:
        -----------
        k : float
            Wavevector outside sphere
        diameter : float
            Sphere diameter
        epsr : complex
            Relative dielectric constant
        mur : float
            Relative magnetic permeability
        l : np.ndarray
            Angular momentum components
            
        Returns:
        --------
        a, b, c, d : np.ndarray
            Mie coefficients
        """
        # Refractive index
        nr = np.sqrt(epsr * mur)
        
        # Riccati-Bessel functions
        j1, _, zjp1, _ = self._riccati_bessel(nr * k * diameter / 2, l)
        j2, h2, zjp2, zhp2 = self._riccati_bessel(k * diameter / 2, l)
        
        # Mie coefficients for outside field
        a = (nr**2 * j1 * zjp2 - mur * j2 * zjp1) / (nr**2 * j1 * zhp2 - mur * h2 * zjp1)
        b = (mur * j1 * zjp2 - j2 * zjp1) / (mur * j1 * zhp2 - h2 * zjp1)
        
        # Mie coefficients for inside field
        c = (mur * j2 * zhp2 - mur * h2 * zjp2) / (mur * j1 * zhp2 - h2 * zjp1)
        d = (mur * nr * j2 * zhp2 - mur * nr * h2 * zjp2) / (nr**2 * j1 * zhp2 - mur * h2 * zjp1)
        
        return a, b, c, d
    
    def scattering(self, enei: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Scattering cross section.
        
        Parameters:
        -----------
        enei : float or np.ndarray
            Wavelength of light in vacuum
            
        Returns:
        --------
        float or np.ndarray
            Scattering cross section
        """
        # Background dielectric function
        epsb = self.epsout(0)
        nb = np.sqrt(epsb)
        
        # Wave vector
        k = 2 * np.pi / enei * nb
        
        # Ratio of dielectric functions
        epsz = self.epsin(enei) / epsb
        
        # Initialize result
        sca = np.zeros_like(enei)
        enei_array = np.atleast_1d(enei)
        k_array = np.atleast_1d(k)
        epsz_array = np.atleast_1d(epsz)
        
        # Unique angular momentum degrees
        l_unique = np.unique(self.ltab)
        
        for i, (ki, epszi) in enumerate(zip(k_array, epsz_array)):
            # Mie coefficients
            a, b, _, _ = self._mie_coefficients(ki, self.diameter, epszi, 1.0, l_unique)
            
            # Scattering cross section
            sca_val = 2 * np.pi / ki**2 * np.sum((2 * l_unique + 1) * (np.abs(a)**2 + np.abs(b)**2))
            
            if np.isscalar(enei):
                return sca_val
            else:
                sca[i] = sca_val
        
        return sca
    
    def extinction(self, enei: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Extinction cross section.
        
        Parameters:
        -----------
        enei : float or np.ndarray
            Wavelength of light in vacuum
            
        Returns:
        --------
        float or np.ndarray
            Extinction cross section
        """
        # Background dielectric function
        epsb = self.epsout(0)
        nb = np.sqrt(epsb)
        
        # Wave vector
        k = 2 * np.pi / enei * nb
        
        # Ratio of dielectric functions
        epsz = self.epsin(enei) / epsb
        
        # Initialize result
        ext = np.zeros_like(enei)
        enei_array = np.atleast_1d(enei)
        k_array = np.atleast_1d(k)
        epsz_array = np.atleast_1d(epsz)
        
        # Unique angular momentum degrees
        l_unique = np.unique(self.ltab)
        
        for i, (ki, epszi) in enumerate(zip(k_array, epsz_array)):
            # Mie coefficients
            a, b, _, _ = self._mie_coefficients(ki, self.diameter, epszi, 1.0, l_unique)
            
            # Extinction cross section
            ext_val = 2 * np.pi / ki**2 * np.sum((2 * l_unique + 1) * np.real(a + b))
            
            if np.isscalar(enei):
                return ext_val
            else:
                ext[i] = ext_val
        
        return ext
    
    def absorption(self, enei: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Absorption cross section.
        
        Parameters:
        -----------
        enei : float or np.ndarray
            Wavelength of light in vacuum
            
        Returns:
        --------
        float or np.ndarray
            Absorption cross section
        """
        return self.extinction(enei) - self.scattering(enei)
    
    def decayrate(self, enei: float, z: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Total and radiative decay rate for oscillating dipole.
        
        Scattering rates are given in units of the free-space decay rate.
        Based on Kim et al., Surf. Science 195, 1 (1988).
        
        Parameters:
        -----------
        enei : float
            Wavelength of light in vacuum (single value only)
        z : float or np.ndarray
            Dipole positions on z axis
            
        Returns:
        --------
        tot : np.ndarray
            Total scattering rate for dipole orientations x and z
        rad : np.ndarray
            Radiative scattering rate for dipole orientations x and z
        """
        assert np.isscalar(enei), "Only single wavelength supported for decayrate"
        
        # Dielectric functions
        epsb = self.epsout(enei)
        k = 2 * np.pi / enei * np.sqrt(epsb)
        
        epsin_val = self.epsin(enei)
        kin = 2 * np.pi / enei * np.sqrt(epsin_val)
        
        z_array = np.atleast_1d(z)
        
        # Initialize results
        tot = np.zeros((len(z_array), 2))  # [x-orientation, z-orientation]
        rad = np.zeros((len(z_array), 2))
        
        # Unique spherical harmonic degrees
        l_unique = np.unique(self.ltab)
        
        # Compute Riccati-Bessel functions for Mie coefficients
        j1, h1, zjp1, zhp1 = self._riccati_bessel(0.5 * k * self.diameter, l_unique)
        j2, _, zjp2, _ = self._riccati_bessel(0.5 * kin * self.diameter, l_unique)
        
        # Modified Mie coefficients [Eq. (11)]
        A = (j1 * zjp2 - j2 * zjp1) / (j2 * zhp1 - h1 * zjp2)
        B = (epsb * j1 * zjp2 - epsin_val * j2 * zjp1) / (epsin_val * j2 * zhp1 - epsb * h1 * zjp2)
        
        # Loop over dipole positions
        for iz, zi in enumerate(z_array):
            # Background wavenumber * dipole position
            y = k * zi
            
            # Spherical Bessel and Hankel functions at dipole position
            j, h, zjp, zhp = self._riccati_bessel(y, l_unique)
            
            # Normalized nonradiative decay rates [Eq. (17,19)]
            tot[iz, 0] = 1 + 1.5 * np.real(np.sum((l_unique + 0.5) * (B * (zhp / y)**2 + A * h**2)))
            tot[iz, 1] = 1 + 1.5 * np.real(np.sum((2 * l_unique + 1) * l_unique * (l_unique + 1) * B * (h / y)**2))
            
            # Normalized radiative decay rates [Eq. (18,20)]
            rad[iz, 0] = 0.75 * np.sum((2 * l_unique + 1) * (np.abs(j + A * h)**2 + np.abs((zjp + B * zhp) / y)**2))
            rad[iz, 1] = 1.5 * np.sum((2 * l_unique + 1) * l_unique * (l_unique + 1) * np.abs((j + B * h) / y)**2)
        
        if np.isscalar(z):
            return tot[0], rad[0]
        else:
            return tot, rad
    
    def loss(self, b: Union[float, np.ndarray], enei: Union[float, np.ndarray], 
            beta: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Energy loss probability for fast electron in vicinity of dielectric sphere.
        
        Based on F. J. Garcia de Abajo, Phys. Rev. B 59, 3095 (1999).
        
        Parameters:
        -----------
        b : float or np.ndarray
            Impact parameter
        enei : float or np.ndarray
            Wavelength of light in vacuum
        beta : float, optional
            Electron velocity in units of speed of light (default: 0.7)
            
        Returns:
        --------
        prob : np.ndarray
            EELS probability, see Eq. (29)
        prad : np.ndarray
            Photon emission probability, Eq. (37)
        """
        # Ensure impact parameter doesn't penetrate sphere
        b_array = np.atleast_1d(b)
        assert np.all(b_array > 0), "Impact parameter must be positive"
        
        # Add sphere radius to impact parameter
        b_eff = 0.5 * self.diameter + b_array
        
        # Gamma factor
        gamma = 1.0 / np.sqrt(1 - beta**2)
        
        enei_array = np.atleast_1d(enei)
        
        # Initialize results
        prob = np.zeros((len(b_eff), len(enei_array)))
        prad = np.zeros((len(b_eff), len(enei_array)))
        
        # Create full spherical harmonics table
        lmax = np.max(self.ltab)
        ltab_full, mtab_full = self._create_sph_table_full(lmax)
        
        # EELS coefficients
        ce, cm = self._aeels(ltab_full, mtab_full, beta)
        
        for ien, enei_val in enumerate(enei_array):
            # Wavenumber in medium
            epsb = self.epsout(enei_val)
            k = 2 * np.pi / enei_val * np.sqrt(epsb)
            
            # Check assumption
            assert epsb == 1, "Mie expressions only implemented for epsb = 1"
            
            # Relative dielectric function
            epsz = self.epsin(enei_val) / epsb
            
            # Mie coefficients
            te, tm, _, _ = self._mie_coefficients(k, self.diameter, epsz, 1.0, ltab_full)
            
            # Correct for different prefactors
            te = 1j * te
            tm = 1j * tm
            
            for ib, bi in enumerate(b_eff):
                # Bessel function
                K = kv(mtab_full, k * bi / (beta * gamma))
                
                # Energy loss probability [Eq. (29)]
                prob[ib, ien] = np.sum(K**2 * (cm * np.imag(tm) + ce * np.imag(te))) / k
                
                # Photon loss probability [Eq. (37)]
                prad[ib, ien] = np.sum(K**2 * (cm * np.abs(tm)**2 + ce * np.abs(te)**2)) / k
        
        # Convert to atomic units (simplified)
        fine = 1/137.036  # fine structure constant
        bohr = 5.29177e-11  # Bohr radius in meters
        hartree = 4.35974e-18  # Hartree energy in J
        
        prob = fine**2 / (bohr * hartree) * prob
        prad = fine**2 / (bohr * hartree) * prad
        
        if np.isscalar(b) and np.isscalar(enei):
            return prob[0, 0], prad[0, 0]
        elif np.isscalar(b):
            return prob[0, :], prad[0, :]
        elif np.isscalar(enei):
            return prob[:, 0], prad[:, 0]
        else:
            return prob, prad
    
    def _create_sph_table_full(self, lmax: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create full spherical harmonics table including m=0."""
        ltab = []
        mtab = []
        
        for l in range(0, lmax + 1):
            for m in range(-l, l + 1):
                ltab.append(l)
                mtab.append(m)
        
        return np.array(ltab), np.array(mtab)
    
    def _aeels(self, ltab: np.ndarray, mtab: np.ndarray, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Spherical harmonics coefficients for EELS.
        
        Based on F. J. Garcia de Abajo, Phys. Rev. B 59, 3095 (1999).
        This is a simplified implementation of the complex calculation.
        """
        # Simplified implementation - in practice this requires complex
        # Legendre polynomial integration and factorial calculations
        gamma = 1.0 / np.sqrt(1 - beta**2)
        
        # Placeholder implementation
        cm = np.ones(len(ltab)) / (ltab * (ltab + 1) + 1e-10)
        ce = np.ones(len(ltab)) / (ltab * (ltab + 1) + 1e-10)
        
        return ce, cm
    
    # Convenient aliases
    def sca(self, enei: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Alias for scattering method."""
        return self.scattering(enei)
    
    def ext(self, enei: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Alias for extinction method."""
        return self.extinction(enei)
    
    def abs(self, enei: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Alias for absorption method."""
        return self.absorption(enei)
    
    def __str__(self):
        """String representation."""
        return f"MieRet: diameter={self.diameter}, lmax={np.max(self.ltab)}"
    
    def __repr__(self):
        """Object representation."""
        return f"MieRet(diameter={self.diameter})"

# Convenience functions
def create_drude_sphere(diameter: float, plasma_freq: float = 1.37e16, 
                       gamma: float = 8.5e13, eps_medium: float = 1.0) -> MieRet:
    """
    Create a sphere with Drude metal dielectric function.
    
    Parameters:
    -----------
    diameter : float
        Sphere diameter
    plasma_freq : float
        Plasma frequency in rad/s
    gamma : float
        Damping frequency in rad/s
    eps_medium : float
        Medium dielectric constant
        
    Returns:
    --------
    MieRet
        Mie theory object for Drude sphere
    """
    def drude_eps(wavelength):
        c = 2.998e8
        omega = 2 * np.pi * c / wavelength
        return 1 - plasma_freq**2 / (omega**2 + 1j * gamma * omega)
    
    def medium_eps(wavelength):
        return eps_medium
    
    return MieRet(drude_eps, medium_eps, diameter)

def compute_mie_spectrum(mie_obj: MieRet, wavelengths: np.ndarray) -> dict:
    """
    Compute full Mie spectrum.
    
    Parameters:
    -----------
    mie_obj : MieRet
        Mie theory object
    wavelengths : np.ndarray
        Wavelength array
        
    Returns:
    --------
    dict
        Spectrum data with wavelengths, scattering, absorption, extinction
    """
    return {
        'wavelengths': wavelengths,
        'scattering': mie_obj.scattering(wavelengths),
        'absorption': mie_obj.absorption(wavelengths),
        'extinction': mie_obj.extinction(wavelengths)
    }