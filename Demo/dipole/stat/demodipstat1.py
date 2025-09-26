"""
DEMODIPSTAT1 - Lifetime reduction for dipole above metallic nanosphere.

For a metallic nanosphere and an oscillating dipole located above the
sphere, with a dipole moment oriented along x or z, this program
computes the total and radiative dipole scattering rates within the
quasistatic approximation, and compares the results with Mie theory.

Runtime: ~3 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import trisphere, comparticle
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
    
    # Initialize sphere
    p = comparticle(epstab, [trisphere(144, diameter)], [2, 1], 1, op)
    
    # Dipole oscillator
    enei = 550
    
    # Positions of dipole
    z = np.linspace(0.6, 1.5, 51) * diameter
    z = z.reshape(-1, 1)
    
    # Compoint
    pt = compoint(p, np.column_stack([np.zeros_like(z), np.zeros_like(z), z]))
    
    # Dipole excitation
    dip = dipole(pt, np.array([[1, 0, 0], [0, 0, 1]]), op)
    
    print("Running quasistatic BEM simulation...")
    
    # BEM simulation
    # Set up BEM solver
    bem = bemsolver(p, op)
    
    # Surface charge
    sig = bem.solve(dip(p, enei))
    
    # Total and radiative decay rate
    tot, rad = dip.decayrate(sig)
    
    print("Quasistatic BEM simulation completed!")
    
    # Final plot
    plt.figure(figsize=(12, 8))
    
    # Plot quasistatic BEM results
    plt.semilogy(z.flatten(), tot[:, 0], '-', label='tot(x) @ BEM', linewidth=2)
    plt.semilogy(z.flatten(), tot[:, 1], '-', label='tot(z) @ BEM', linewidth=2)
    plt.semilogy(z.flatten(), rad[:, 0], 'o-', label='rad(x) @ BEM', markersize=4)
    plt.semilogy(z.flatten(), rad[:, 1], 'o-', label='rad(z) @ BEM', markersize=4)
    
    plt.title('Total and radiative decay rate for dipole oriented along x and z')
    plt.xlabel('Position (nm)')
    plt.ylabel('Decay rate')
    
    print("Computing Mie theory comparison...")
    
    # Comparison with Mie theory
    mie = miesolver(epstab[1], epstab[0], diameter, op)
    
    # Total and radiative decay rate
    tot0, rad0 = mie.decayrate(enei, z.flatten())
    
    # Plot Mie theory results
    plt.semilogy(z.flatten(), tot0[:, 0], '--', label='tot(x) @ Mie', linewidth=2)
    plt.semilogy(z.flatten(), tot0[:, 1], '--', label='tot(z) @ Mie', linewidth=2)
    plt.semilogy(z.flatten(), rad0[:, 0], 'o--', label='rad(x) @ Mie', markersize=4)
    plt.semilogy(z.flatten(), rad0[:, 1], 'o--', label='rad(z) @ Mie', markersize=4)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("Mie theory comparison completed!")
    
    # Additional analysis
    analyze_quasistatic_validity(tot, rad, tot0, rad0, z, diameter)
    
    return tot, rad, tot0, rad0, z


def analyze_quasistatic_validity(tot, rad, tot0, rad0, z, diameter):
    """Analyze validity of quasistatic approximation"""
    
    print("\n=== Quasistatic Approximation Analysis ===")
    
    # Calculate relative differences between quasistatic BEM and Mie
    rel_diff_tot_x = np.abs(tot[:, 0] - tot0[:, 0]) / tot0[:, 0] * 100
    rel_diff_tot_z = np.abs(tot[:, 1] - tot0[:, 1]) / tot0[:, 1] * 100
    rel_diff_rad_x = np.abs(rad[:, 0] - rad0[:, 0]) / rad0[:, 0] * 100
    rel_diff_rad_z = np.abs(rad[:, 1] - rad0[:, 1]) / rad0[:, 1] * 100
    
    print("Quasistatic BEM vs Mie theory agreement:")
    print(f"Total decay rate (x): Mean diff = {np.mean(rel_diff_tot_x):.2f}%, Max diff = {np.max(rel_diff_tot_x):.2f}%")
    print(f"Total decay rate (z): Mean diff = {np.mean(rel_diff_tot_z):.2f}%, Max diff = {np.max(rel_diff_tot_z):.2f}%")
    print(f"Radiative decay rate (x): Mean diff = {np.mean(rel_diff_rad_x):.2f}%, Max diff = {np.max(rel_diff_rad_x):.2f}%")
    print(f"Radiative decay rate (z): Mean diff = {np.mean(rel_diff_rad_z):.2f}%, Max diff = {np.max(rel_diff_rad_z):.2f}%")
    
    # Size parameter analysis
    wavelength = 550  # nm
    size_parameter = np.pi * diameter / wavelength
    
    print(f"\nSize parameter analysis:")
    print(f"Particle diameter: {diameter} nm")
    print(f"Wavelength: {wavelength} nm")
    print(f"Size parameter (πD/λ): {size_parameter:.3f}")
    
    if size_parameter < 0.1:
        print("Size parameter << 1: Quasistatic approximation is excellent")
    elif size_parameter < 0.3:
        print("Size parameter < 0.3: Quasistatic approximation is good")
    else:
        print("Size parameter > 0.3: Quasistatic approximation may have limitations")
    
    # Distance analysis
    min_distance = z[0, 0] - diameter/2
    max_distance = z[-1, 0] - diameter/2
    
    print(f"\nDipole distance from surface:")
    print(f"Minimum: {min_distance:.1f} nm")
    print(f"Maximum: {max_distance:.1f} nm")
    
    # Find position of maximum enhancement
    max_tot_x_idx = np.argmax(tot[:, 0])
    max_tot_z_idx = np.argmax(tot[:, 1])
    
    print(f"\nMaximum enhancements (quasistatic):")
    print(f"x-polarization: {tot[max_tot_x_idx, 0]:.2f} at z = {z[max_tot_x_idx, 0]:.1f} nm")
    print(f"z-polarization: {tot[max_tot_z_idx, 0]:.2f} at z = {z[max_tot_z_idx, 0]:.1f} nm")


def plot_detailed_comparison(tot, rad, tot0, rad0, z):
    """Detailed comparison plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total decay rates
    axes[0, 0].semilogy(z.flatten(), tot[:, 0], 'r-', label='Quasistatic BEM', linewidth=2)
    axes[0, 0].semilogy(z.flatten(), tot0[:, 0], 'r--', label='Mie theory', linewidth=2)
    axes[0, 0].set_xlabel('Position (nm)')
    axes[0, 0].set_ylabel('Total decay rate')
    axes[0, 0].set_title('x-polarized dipole')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].semilogy(z.flatten(), tot[:, 1], 'b-', label='Quasistatic BEM', linewidth=2)
    axes[0, 1].semilogy(z.flatten(), tot0[:, 1], 'b--', label='Mie theory', linewidth=2)
    axes[0, 1].set_xlabel('Position (nm)')
    axes[0, 1].set_ylabel('Total decay rate')
    axes[0, 1].set_title('z-polarized dipole')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Radiative decay rates
    axes[1, 0].semilogy(z.flatten(), rad[:, 0], 'r-', label='Quasistatic BEM', linewidth=2)
    axes[1, 0].semilogy(z.flatten(), rad0[:, 0], 'r--', label='Mie theory', linewidth=2)
    axes[1, 0].set_xlabel('Position (nm)')
    axes[1, 0].set_ylabel('Radiative decay rate')
    axes[1, 0].set_title('x-polarized dipole (radiative)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].semilogy(z.flatten(), rad[:, 1], 'b-', label='Quasistatic BEM', linewidth=2)
    axes[1, 1].semilogy(z.flatten(), rad0[:, 1], 'b--', label='Mie theory', linewidth=2)
    axes[1, 1].set_xlabel('Position (nm)')
    axes[1, 1].set_ylabel('Radiative decay rate')
    axes[1, 1].set_title('z-polarized dipole (radiative)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Relative difference plot
    plt.figure(figsize=(12, 6))
    
    rel_diff_tot_x = (tot[:, 0] - tot0[:, 0]) / tot0[:, 0] * 100
    rel_diff_tot_z = (tot[:, 1] - tot0[:, 1]) / tot0[:, 1] * 100
    
    plt.subplot(1, 2, 1)
    plt.plot(z.flatten(), rel_diff_tot_x, 'r-', label='x-polarization', linewidth=2)
    plt.plot(z.flatten(), rel_diff_tot_z, 'b-', label='z-polarization', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Position (nm)')
    plt.ylabel('Relative difference (%)')
    plt.title('Total decay rate: (Quasistatic - Mie)/Mie')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Non-radiative contribution
    nonrad_qs = tot - rad
    nonrad_mie = tot0 - rad0
    
    plt.subplot(1, 2, 2)
    plt.semilogy(z.flatten(), nonrad_qs[:, 0], 'r-', label='x-pol (QS)', linewidth=2)
    plt.semilogy(z.flatten(), nonrad_qs[:, 1], 'b-', label='z-pol (QS)', linewidth=2)
    plt.semilogy(z.flatten(), nonrad_mie[:, 0], 'r--', label='x-pol (Mie)', linewidth=2)
    plt.semilogy(z.flatten(), nonrad_mie[:, 1], 'b--', label='z-pol (Mie)', linewidth=2)
    plt.xlabel('Position (nm)')
    plt.ylabel('Non-radiative decay rate')
    plt.title('Non-radiative decay rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    tot, rad, tot0, rad0, z = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Gold nanosphere: {10} nm diameter")
    print(f"Dipole positions: z = {z[0, 0]:.1f} - {z[-1, 0]:.1f} nm")
    print(f"Number of positions: {len(z)}")
    print(f"Wavelength: 550 nm")
    print(f"Method: Quasistatic BEM vs Mie theory")
    print(f"Particle size regime: Small compared to wavelength")
    
    # Show detailed comparison
    plot_detailed_comparison(tot, rad, tot0, rad0, z)