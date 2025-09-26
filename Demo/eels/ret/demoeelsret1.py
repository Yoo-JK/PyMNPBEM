"""
DEMOEELSRET1 - Comparison BEM and Mie for EELS of metallic nanosphere.

For a metallic nanosphere of 150 nm, this program computes the energy
loss probability for an impact parameter of 20 nm and for different
loss energies using the full Maxwell equations, and compares the
results with Mie theory.

See also F. J. Garcia de Abajo, Phys. Rev. B 59, 3095 (1999).

Runtime: ~18.7 seconds.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import trisphere, comparticle
from pymnpbem.misc import units, eelsbase
from pymnpbem.bem import bemsolver, electronbeam
from pymnpbem.mie import miesolver


def main():
    # Initialization
    # Options for BEM simulation
    op = bemoptions(sim='ret', interp='curv')
    
    # Table of dielectric function
    epstab = [epsconst(1), epstable('silver.dat')]
    
    # Diameter
    diameter = 150
    
    # Nanosphere
    p = comparticle(epstab, [trisphere(256, diameter)], [2, 1], 1, op)
    
    # Width of electron beam and electron velocity
    width = 0.5
    vel = eelsbase.ene2vel(200e3)  # 200 keV electron energy
    
    # Impact parameter
    imp = 20
    
    # Loss energies in eV
    ene = np.linspace(1.5, 4.5, 40)
    
    # Convert energies to nm
    unit_converter = units()
    enei = unit_converter.eV2nm / ene
    
    # BEM solution
    # BEM solver
    bem = bemsolver(p, op)
    
    # Electron beam excitation
    exc = electronbeam(p, [diameter / 2 + imp, 0], width, vel, op)
    
    # Surface loss
    psurf = np.zeros_like(ene)
    
    print("Running BEM simulation for EELS...")
    
    # Loop over energies
    for ien in tqdm(range(len(enei)), desc="BEM solver", ncols=80):
        # Surface charges
        sig = bem.solve(exc(enei[ien]))
        
        # EELS losses
        psurf[ien] = exc.loss(sig)
    
    print("BEM EELS simulation completed!")
    
    # Mie solver
    mie = miesolver(epstab[1], epstab[0], diameter, op, lmax=40)
    
    # Mie theory EELS calculation
    pmie = mie.loss(imp, enei, vel)
    
    # Final plot
    plt.figure(figsize=(12, 8))
    
    plt.plot(ene, psurf, 'o-', label='BEM', linewidth=2, markersize=6)
    plt.plot(ene, pmie, '.-', label='Mie', linewidth=2, markersize=6)
    
    plt.legend()
    plt.xlabel('Loss energy (eV)')
    plt.ylabel('Loss probability (eV⁻¹)')
    plt.title(f'EELS of silver nanosphere (d={diameter}nm, impact={imp}nm)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    analyze_eels_results(ene, psurf, pmie, diameter, imp, vel)
    
    return psurf, pmie, ene


def analyze_eels_results(ene, psurf, pmie, diameter, imp, vel):
    """Analyze EELS results and plasmon resonances"""
    
    print("\n=== EELS Analysis ===")
    
    # Find plasmon resonance peaks
    bem_peaks = find_peaks(ene, psurf)
    mie_peaks = find_peaks(ene, pmie)
    
    print("BEM plasmon peaks:")
    for i, (energy, intensity) in enumerate(bem_peaks):
        print(f"  Peak {i+1}: {energy:.2f} eV, Intensity = {intensity:.3e} eV⁻¹")
    
    print("Mie theory peaks:")
    for i, (energy, intensity) in enumerate(mie_peaks):
        print(f"  Peak {i+1}: {energy:.2f} eV, Intensity = {intensity:.3e} eV⁻¹")
    
    # Compare methods
    relative_diff = np.abs(psurf - pmie) / (pmie + 1e-15) * 100
    mean_diff = np.mean(relative_diff)
    max_diff = np.max(relative_diff)
    
    print(f"\nBEM vs Mie comparison:")
    print(f"Mean relative difference: {mean_diff:.1f}%")
    print(f"Maximum relative difference: {max_diff:.1f}%")
    
    # EELS parameters
    print(f"\nEELS parameters:")
    print(f"Electron velocity: {vel:.2e} m/s ({vel/2.998e8*100:.1f}% of light speed)")
    print(f"Impact parameter: {imp} nm")
    print(f"Particle diameter: {diameter} nm")
    print(f"Energy range: {ene[0]:.1f} - {ene[-1]:.1f} eV")
    
    # Maximum loss probability
    max_bem = np.max(psurf)
    max_mie = np.max(pmie)
    max_energy_bem = ene[np.argmax(psurf)]
    max_energy_mie = ene[np.argmax(pmie)]
    
    print(f"\nMaximum loss probabilities:")
    print(f"BEM: {max_bem:.3e} eV⁻¹ at {max_energy_bem:.2f} eV")
    print(f"Mie: {max_mie:.3e} eV⁻¹ at {max_energy_mie:.2f} eV")


def find_peaks(energy, intensity, threshold_factor=0.1):
    """Find peaks in EELS spectrum"""
    
    peaks = []
    max_intensity = np.max(intensity)
    threshold = threshold_factor * max_intensity
    
    for i in range(1, len(energy)-1):
        if (intensity[i] > intensity[i-1] and 
            intensity[i] > intensity[i+1] and 
            intensity[i] > threshold):
            peaks.append((energy[i], intensity[i]))
    
    return peaks


def plot_detailed_eels_analysis(ene, psurf, pmie):
    """Detailed EELS analysis plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Linear scale comparison
    axes[0, 0].plot(ene, psurf, 'r-o', label='BEM', linewidth=2, markersize=4)
    axes[0, 0].plot(ene, pmie, 'b-s', label='Mie', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Loss energy (eV)')
    axes[0, 0].set_ylabel('Loss probability (eV⁻¹)')
    axes[0, 0].set_title('EELS Spectra Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Log scale comparison
    axes[0, 1].semilogy(ene, psurf, 'r-o', label='BEM', linewidth=2, markersize=4)
    axes[0, 1].semilogy(ene, pmie, 'b-s', label='Mie', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Loss energy (eV)')
    axes[0, 1].set_ylabel('Loss probability (eV⁻¹)')
    axes[0, 1].set_title('EELS Spectra (Log Scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Relative difference
    rel_diff = (psurf - pmie) / (pmie + 1e-15) * 100
    axes[1, 0].plot(ene, rel_diff, 'g-', linewidth=2)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Loss energy (eV)')
    axes[1, 0].set_ylabel('Relative difference (%)')
    axes[1, 0].set_title('(BEM - Mie)/Mie × 100%')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Normalized comparison
    psurf_norm = psurf / np.max(psurf)
    pmie_norm = pmie / np.max(pmie)
    
    axes[1, 1].plot(ene, psurf_norm, 'r-', label='BEM (normalized)', linewidth=2)
    axes[1, 1].plot(ene, pmie_norm, 'b--', label='Mie (normalized)', linewidth=2)
    axes[1, 1].set_xlabel('Loss energy (eV)')
    axes[1, 1].set_ylabel('Normalized loss probability')
    axes[1, 1].set_title('Normalized EELS Spectra')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def explain_eels_physics():
    """Explain the physics of EELS"""
    
    print("\n=== EELS Physics ===")
    print("Electron Energy Loss Spectroscopy (EELS):")
    print("- Fast electron passes near nanoparticle")
    print("- Electron's electromagnetic field excites plasmons")
    print("- Electron loses energy equal to plasmon energy")
    print("- Loss probability depends on impact parameter")
    print("- Surface plasmons dominate for metallic nanoparticles")
    print("- BEM captures retardation effects vs Mie theory")


if __name__ == '__main__':
    psurf, pmie, ene = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Silver nanosphere: 150 nm diameter")
    print(f"Electron beam: 200 keV energy")
    print(f"Impact parameter: 20 nm")
    print(f"Loss energy range: {ene[0]:.1f} - {ene[-1]:.1f} eV")
    print(f"Methods: BEM (retarded) vs Mie theory")
    
    # Detailed analysis
    plot_detailed_eels_analysis(ene, psurf, pmie)
    explain_eels_physics()