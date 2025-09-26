"""
DEMODIPRET6 - Lifetime reduction for dipole above a coated nanosphere.

For a coated metallic nanosphere (50 nm glass sphere coated with 10 nm
thick silver shell) and an oscillating dipole located above the
sphere, with a dipole moment oriented along x or z, this program
computes the total and radiative dipole scattering rates for different
dipole transition energies using the full Maxwell equations.

Runtime: ~67 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import trisphere, comparticle
from pymnpbem.misc import compoint
from pymnpbem.bem import dipole, bemsolver


def main():
    # Initialization
    # Options for BEM simulation
    op = bemoptions(sim='ret', interp='curv')
    
    # Table of dielectric functions
    # 1: vacuum/air, 2: gold (shell), 3: glass (core)
    epstab = [epsconst(1), epstable('gold.dat'), epsconst(2.25)]
    
    # Diameter of sphere
    diameter = 50
    
    # Nanospheres
    p1 = trisphere(144, diameter)        # Inner sphere (glass core)
    p2 = trisphere(256, diameter + 10)   # Outer sphere (gold shell)
    
    # Initialize sphere
    # Material indices: [3, 2; 2, 1] means:
    # Inner sphere: glass(3) inside, gold(2) outside
    # Outer sphere: gold(2) inside, air(1) outside
    p = comparticle(epstab, [p1, p2], [[3, 2], [2, 1]], 1, 2, op)
    
    # Dipole oscillator
    enei = np.linspace(400, 900, 40)
    
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
    
    print("Running BEM simulation for coated nanosphere...")
    
    # Loop over wavelengths
    for ien in tqdm(range(len(enei)), desc="BEM solver", ncols=80):
        # Surface charge
        sig = bem.solve(dip(p, enei[ien]))
        
        # Total and radiative decay rate
        tot[ien, :], rad[ien, :] = dip.decayrate(sig)
    
    print("BEM simulation completed!")
    
    # Final plot
    plt.figure(figsize=(12, 8))
    
    # Plot total decay rates
    plt.plot(enei, tot[:, 0], '-', label='tot(x)', linewidth=2)
    plt.plot(enei, tot[:, 1], '-', label='tot(z)', linewidth=2)
    
    # Plot radiative decay rates
    plt.plot(enei, rad[:, 0], 'o-', label='rad(x)', markersize=4)
    plt.plot(enei, rad[:, 1], 'o-', label='rad(z)', markersize=4)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Decay rate')
    plt.legend()
    plt.title('Decay rates for dipole above coated nanosphere (50nm glass + 10nm gold)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Additional analysis plots
    plot_analysis(enei, tot, rad)
    
    return tot, rad, enei


def plot_analysis(enei, tot, rad):
    """Additional analysis plots"""
    
    # Calculate non-radiative decay rates
    nonrad = tot - rad
    
    # Plot decay rate components
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # x-polarization
    axes[0, 0].plot(enei, tot[:, 0], 'b-', label='Total', linewidth=2)
    axes[0, 0].plot(enei, rad[:, 0], 'r-', label='Radiative', linewidth=2)
    axes[0, 0].plot(enei, nonrad[:, 0], 'g-', label='Non-radiative', linewidth=2)
    axes[0, 0].set_xlabel('Wavelength (nm)')
    axes[0, 0].set_ylabel('Decay rate')
    axes[0, 0].set_title('x-polarized dipole')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # z-polarization
    axes[0, 1].plot(enei, tot[:, 1], 'b-', label='Total', linewidth=2)
    axes[0, 1].plot(enei, rad[:, 1], 'r-', label='Radiative', linewidth=2)
    axes[0, 1].plot(enei, nonrad[:, 1], 'g-', label='Non-radiative', linewidth=2)
    axes[0, 1].set_xlabel('Wavelength (nm)')
    axes[0, 1].set_ylabel('Decay rate')
    axes[0, 1].set_title('z-polarized dipole')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Quantum efficiency
    qe_x = rad[:, 0] / tot[:, 0]
    qe_z = rad[:, 1] / tot[:, 1]
    
    axes[1, 0].plot(enei, qe_x, 'b-', label='x-polarization', linewidth=2)
    axes[1, 0].plot(enei, qe_z, 'r-', label='z-polarization', linewidth=2)
    axes[1, 0].set_xlabel('Wavelength (nm)')
    axes[1, 0].set_ylabel('Quantum efficiency')
    axes[1, 0].set_title('Radiative quantum efficiency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Enhancement factors (compared to free space)
    # Assuming free space decay rate = 1 (normalized)
    enhancement_tot_x = tot[:, 0]
    enhancement_tot_z = tot[:, 1]
    
    axes[1, 1].plot(enei, enhancement_tot_x, 'b-', label='Total (x)', linewidth=2)
    axes[1, 1].plot(enei, enhancement_tot_z, 'r-', label='Total (z)', linewidth=2)
    axes[1, 1].axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Free space')
    axes[1, 1].set_xlabel('Wavelength (nm)')
    axes[1, 1].set_ylabel('Enhancement factor')
    axes[1, 1].set_title('Total decay rate enhancement')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_coated_sphere_response(enei, tot, rad):
    """Analyze spectral response of coated sphere"""
    
    print("\n=== Coated Sphere Analysis ===")
    
    # Find resonance peaks
    for pol, label in enumerate(['x', 'z']):
        print(f"\n{label.upper()}-polarization:")
        
        # Find local maxima in total decay rate
        peaks = []
        for i in range(1, len(enei)-1):
            if tot[i, pol] > tot[i-1, pol] and tot[i, pol] > tot[i+1, pol]:
                if tot[i, pol] > 2.0:  # Only significant peaks
                    peaks.append((enei[i], tot[i, pol], rad[i, pol]))
        
        if peaks:
            for j, (wavelength, tot_val, rad_val) in enumerate(peaks):
                qe = rad_val / tot_val
                print(f"  Peak {j+1}: λ = {wavelength:.0f} nm")
                print(f"    Total decay: {tot_val:.2f}, Radiative: {rad_val:.2f}")
                print(f"    Quantum efficiency: {qe:.3f}")
        
        # Overall statistics
        max_tot_idx = np.argmax(tot[:, pol])
        max_rad_idx = np.argmax(rad[:, pol])
        
        print(f"  Maximum total decay: {tot[max_tot_idx, pol]:.2f} at {enei[max_tot_idx]:.0f} nm")
        print(f"  Maximum radiative decay: {rad[max_rad_idx, pol]:.2f} at {enei[max_rad_idx]:.0f} nm")
        
        # Average quantum efficiency
        avg_qe = np.mean(rad[:, pol] / tot[:, pol])
        print(f"  Average quantum efficiency: {avg_qe:.3f}")


if __name__ == '__main__':
    tot, rad, enei = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Wavelength range: {enei[0]:.0f} - {enei[-1]:.0f} nm")
    print(f"Core: 50 nm diameter glass sphere (n = {np.sqrt(2.25):.2f})")
    print(f"Shell: 10 nm thick gold coating")
    print(f"Dipole position: 35 nm above surface (z = 35 nm)")
    
    # Analyze spectral response
    analyze_coated_sphere_response(enei, tot, rad)