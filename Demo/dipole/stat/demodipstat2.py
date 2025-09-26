"""
DEMODIPSTAT2 - Energy dependent lifetime for dipole above a nanosphere.

For a metallic nanosphere and an oscillating dipole located above the
sphere, with a dipole moment oriented along x or z, this program
computes the total and radiative dipole scattering rates for different
dipole transition energies within the quasistatic approximation, and
compares the results with Mie theory.

Runtime: ~12 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import trispheresegment, comparticle
from pymnpbem.misc import compoint
from pymnpbem.bem import dipole, bemsolver
from pymnpbem.mie import miesolver


def main():
    # Initialization
    # Options for BEM simulation (quasistatic)
    op = bemoptions(sim='stat', waitbar=0, interp='curv')
    
    # Table of dielectric functions
    epstab = [epsconst(1), epstable('gold.dat')]
    
    # Diameter of sphere
    diameter = 10
    
    # Nanosphere with finer discretization at the top
    phi_grid = 2 * np.pi * np.linspace(0, 1, 31)
    theta_grid = np.pi * np.linspace(0, 1, 31) ** 2
    p_mesh = trispheresegment(phi_grid, theta_grid, diameter)
    
    # Initialize sphere
    p = comparticle(epstab, [p_mesh], [2, 1], 1, op)
    
    # Dipole oscillator
    enei = np.linspace(400, 600, 40)
    
    # Compoint
    pt = compoint(p, [0, 0, 0.7 * diameter])
    
    # Dipole excitation
    dip = dipole(pt, np.array([[1, 0, 0], [0, 0, 1]]), op)
    
    # Initialize total and radiative scattering rate
    tot = np.zeros((len(enei), 2))
    rad = np.zeros((len(enei), 2))
    
    # BEM simulation
    # Set up BEM solver
    bem = bemsolver(p, op)
    
    print("Running quasistatic BEM simulation over wavelength range...")
    
    # Loop over wavelengths
    for ien in tqdm(range(len(enei)), desc="BEM solver", ncols=80):
        # Surface charge
        sig = bem.solve(dip(p, enei[ien]))
        
        # Total and radiative decay rate
        tot[ien, :], rad[ien, :] = dip.decayrate(sig)
    
    print("Quasistatic BEM simulation completed!")
    
    # Final plot
    plt.figure(figsize=(12, 8))
    
    # Plot quasistatic BEM results
    plt.plot(enei, tot[:, 0], '-', label='tot(x) @ BEM', linewidth=2)
    plt.plot(enei, tot[:, 1], '-', label='tot(z) @ BEM', linewidth=2)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Total decay rate')
    
    print("Computing Mie theory comparison...")
    
    # Comparison with Mie theory
    mie = miesolver(epstab[1], epstab[0], diameter, op)
    
    # Total and radiative decay rate
    tot0 = np.zeros((len(enei), 2))
    rad0 = np.zeros((len(enei), 2))
    
    # Loop over energies
    for ien in tqdm(range(len(enei)), desc="Mie theory", ncols=80):
        tot0[ien, :], rad0[ien, :] = mie.decayrate(enei[ien], pt.pos[:, 2])  # z-coordinate
    
    # Plot Mie theory results
    plt.plot(enei, tot0[:, 0], 'o', label='tot(x) @ Mie', markersize=4)
    plt.plot(enei, tot0[:, 1], 'o', label='tot(z) @ Mie', markersize=4)
    
    plt.legend()
    plt.title(f'Energy dependent decay rates for dipole at z = {0.7 * diameter:.1f} nm (quasistatic)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("Mie theory comparison completed!")
    
    # Additional analysis
    analyze_spectral_response(enei, tot, rad, tot0, rad0, diameter)
    
    return tot, rad, tot0, rad0, enei


def analyze_spectral_response(enei, tot, rad, tot0, rad0, diameter):
    """Analyze spectral response and plasmon resonances"""
    
    print("\n=== Spectral Response Analysis ===")
    
    # Find resonance peaks in quasistatic calculation
    for pol, label in enumerate(['x', 'z']):
        print(f"\n{label.upper()}-polarization:")
        
        # Find local maxima in total decay rate
        peaks_qs = []
        peaks_mie = []
        
        for i in range(2, len(enei)-2):
            # Quasistatic peaks
            if (tot[i, pol] > tot[i-1, pol] and tot[i, pol] > tot[i+1, pol] and
                tot[i, pol] > tot[i-2, pol] and tot[i, pol] > tot[i+2, pol]):
                if tot[i, pol] > 1.5:  # Only significant peaks
                    peaks_qs.append((enei[i], tot[i, pol], rad[i, pol]))
            
            # Mie theory peaks
            if (tot0[i, pol] > tot0[i-1, pol] and tot0[i, pol] > tot0[i+1, pol] and
                tot0[i, pol] > tot0[i-2, pol] and tot0[i, pol] > tot0[i+2, pol]):
                if tot0[i, pol] > 1.5:  # Only significant peaks
                    peaks_mie.append((enei[i], tot0[i, pol], rad0[i, pol]))
        
        print("  Quasistatic BEM peaks:")
        if peaks_qs:
            for j, (wavelength, tot_val, rad_val) in enumerate(peaks_qs):
                qe = rad_val / tot_val if tot_val > 0 else 0
                print(f"    Peak {j+1}: λ = {wavelength:.0f} nm, Total = {tot_val:.2f}, QE = {qe:.3f}")
        else:
            max_idx = np.argmax(tot[:, pol])
            print(f"    Maximum: λ = {enei[max_idx]:.0f} nm, Total = {tot[max_idx, pol]:.2f}")
        
        print("  Mie theory peaks:")
        if peaks_mie:
            for j, (wavelength, tot_val, rad_val) in enumerate(peaks_mie):
                qe = rad_val / tot_val if tot_val > 0 else 0
                print(f"    Peak {j+1}: λ = {wavelength:.0f} nm, Total = {tot_val:.2f}, QE = {qe:.3f}")
        else:
            max_idx = np.argmax(tot0[:, pol])
            print(f"    Maximum: λ = {enei[max_idx]:.0f} nm, Total = {tot0[max_idx, pol]:.2f}")
    
    # Overall comparison statistics
    rel_diff_tot_x = np.abs(tot[:, 0] - tot0[:, 0]) / (tot0[:, 0] + 1e-10) * 100
    rel_diff_tot_z = np.abs(tot[:, 1] - tot0[:, 1]) / (tot0[:, 1] + 1e-10) * 100
    
    print(f"\nOverall agreement (Quasistatic vs Mie):")
    print(f"Total decay rate (x): Mean diff = {np.mean(rel_diff_tot_x):.1f}%, Max diff = {np.max(rel_diff_tot_x):.1f}%")
    print(f"Total decay rate (z): Mean diff = {np.mean(rel_diff_tot_z):.1f}%, Max diff = {np.max(rel_diff_tot_z):.1f}%")


def plot_detailed_spectral_analysis(enei, tot, rad, tot0, rad0):
    """Detailed spectral analysis plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total decay rates comparison
    axes[0, 0].plot(enei, tot[:, 0], 'r-', label='Quasistatic BEM', linewidth=2)
    axes[0, 0].plot(enei, tot0[:, 0], 'ro', label='Mie theory', markersize=4)
    axes[0, 0].set_xlabel('Wavelength (nm)')
    axes[0, 0].set_ylabel('Total decay rate')
    axes[0, 0].set_title('x-polarized dipole')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(enei, tot[:, 1], 'b-', label='Quasistatic BEM', linewidth=2)
    axes[0, 1].plot(enei, tot0[:, 1], 'bo', label='Mie theory', markersize=4)
    axes[0, 1].set_xlabel('Wavelength (nm)')
    axes[0, 1].set_ylabel('Total decay rate')
    axes[0, 1].set_title('z-polarized dipole')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Radiative vs non-radiative comparison
    nonrad_qs = tot - rad
    nonrad_mie = tot0 - rad0
    
    axes[1, 0].plot(enei, rad[:, 0], 'r-', label='Radiative (QS)', linewidth=2)
    axes[1, 0].plot(enei, nonrad_qs[:, 0], 'r--', label='Non-radiative (QS)', linewidth=2)
    axes[1, 0].plot(enei, rad0[:, 0], 'ro', label='Radiative (Mie)', markersize=3)
    axes[1, 0].plot(enei, nonrad_mie[:, 0], 'r^', label='Non-radiative (Mie)', markersize=3)
    axes[1, 0].set_xlabel('Wavelength (nm)')
    axes[1, 0].set_ylabel('Decay rate')
    axes[1, 0].set_title('x-polarization: Radiative vs Non-radiative')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Quantum efficiency
    qe_qs_x = rad[:, 0] / (tot[:, 0] + 1e-10)
    qe_qs_z = rad[:, 1] / (tot[:, 1] + 1e-10)
    qe_mie_x = rad0[:, 0] / (tot0[:, 0] + 1e-10)
    qe_mie_z = rad0[:, 1] / (tot0[:, 1] + 1e-10)
    
    axes[1, 1].plot(enei, qe_qs_x, 'r-', label='x-pol (QS)', linewidth=2)
    axes[1, 1].plot(enei, qe_qs_z, 'b-', label='z-pol (QS)', linewidth=2)
    axes[1, 1].plot(enei, qe_mie_x, 'ro', label='x-pol (Mie)', markersize=3)
    axes[1, 1].plot(enei, qe_mie_z, 'bo', label='z-pol (Mie)', markersize=3)
    axes[1, 1].set_xlabel('Wavelength (nm)')
    axes[1, 1].set_ylabel('Quantum efficiency')
    axes[1, 1].set_title('Radiative quantum efficiency')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    tot, rad, tot0, rad0, enei = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Gold nanosphere: {10} nm diameter")
    print(f"Dipole position: z = {0.7 * 10:.1f} nm")
    print(f"Wavelength range: {enei[0]:.0f} - {enei[-1]:.0f} nm")
    print(f"Number of wavelength points: {len(enei)}")
    print(f"Method: Quasistatic BEM vs Mie theory")
    print(f"Focus: Spectral response and plasmon resonances")
    
    # Show detailed spectral analysis
    plot_detailed_spectral_analysis(enei, tot, rad, tot0, rad0)