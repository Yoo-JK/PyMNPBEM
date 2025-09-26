"""
DEMODIPRET12 - Dipole lifetime for nanosphere and mirror symmetry.

For a metallic nanosphere and an oscillating dipole located above the
sphere, with a dipole moment oriented along x or z, this program
computes the total and radiative dipole scattering rates using mirror
symmetry and the full Maxwell equations and compares the results with
Mie theory.

Runtime: ~4 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import trispheresegment, comparticlemirror
from pymnpbem.misc import compoint
from pymnpbem.bem import dipole, bemsolver
from pymnpbem.mie import miesolver


def main():
    # Initialization
    # Options for BEM simulation with mirror symmetry
    op = bemoptions(sim='ret', interp='curv', sym='xy')
    
    # Table of dielectric functions
    epstab = [epsconst(1), epstable('gold.dat')]
    
    # Diameter of sphere
    diameter = 150
    
    # One quarter of a sphere, use finer discretization at north pole
    phi_grid = np.linspace(0, np.pi / 2, 10)
    theta_grid = np.pi * np.linspace(0, 1, 15) ** 2
    p = trispheresegment(phi_grid, theta_grid, diameter)
    
    # Initialize sphere with mirror symmetry
    p = comparticlemirror(epstab, [p], [2, 1], 1, op)
    
    # Dipole oscillator
    enei = 550
    
    # Positions of dipole
    z = np.linspace(0.6, 1.5, 51) * diameter
    z = z.reshape(-1, 1)
    
    # Compoint
    pt = compoint(p, np.column_stack([np.zeros_like(z), np.zeros_like(z), z]))
    
    # Dipole excitation
    dip = dipole(pt, np.array([[1, 0, 0], [0, 0, 1]]), op)
    
    print("Running BEM simulation with mirror symmetry...")
    
    # BEM simulation
    # Set up BEM solver
    bem = bemsolver(p, op)
    
    # Surface charge
    sig = bem.solve(dip(p, enei))
    
    # Total and radiative decay rate
    tot, rad = dip.decayrate(sig)
    
    print("BEM simulation completed!")
    
    # Final plot
    plt.figure(figsize=(12, 8))
    
    # Plot BEM results
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
    
    plt.xlim([np.min(z), np.max(z)])
    
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
    analyze_mirror_symmetry_benefits(tot, rad, tot0, rad0, z)
    
    return tot, rad, tot0, rad0, z


def analyze_mirror_symmetry_benefits(tot, rad, tot0, rad0, z):
    """Analyze benefits of using mirror symmetry"""
    
    print("\n=== Mirror Symmetry Analysis ===")
    
    # Calculate relative differences between BEM and Mie
    rel_diff_tot_x = np.abs(tot[:, 0] - tot0[:, 0]) / tot0[:, 0] * 100
    rel_diff_tot_z = np.abs(tot[:, 1] - tot0[:, 1]) / tot0[:, 1] * 100
    rel_diff_rad_x = np.abs(rad[:, 0] - rad0[:, 0]) / rad0[:, 0] * 100
    rel_diff_rad_z = np.abs(rad[:, 1] - rad0[:, 1]) / rad0[:, 1] * 100
    
    print("BEM vs Mie theory agreement:")
    print(f"Total decay rate (x): Mean diff = {np.mean(rel_diff_tot_x):.2f}%, Max diff = {np.max(rel_diff_tot_x):.2f}%")
    print(f"Total decay rate (z): Mean diff = {np.mean(rel_diff_tot_z):.2f}%, Max diff = {np.max(rel_diff_tot_z):.2f}%")
    print(f"Radiative decay rate (x): Mean diff = {np.mean(rel_diff_rad_x):.2f}%, Max diff = {np.max(rel_diff_rad_x):.2f}%")
    print(f"Radiative decay rate (z): Mean diff = {np.mean(rel_diff_rad_z):.2f}%, Max diff = {np.max(rel_diff_rad_z):.2f}%")
    
    # Find positions of maximum enhancement
    max_tot_x_idx = np.argmax(tot[:, 0])
    max_tot_z_idx = np.argmax(tot[:, 1])
    
    print(f"\nMaximum enhancements:")
    print(f"x-polarization: {tot[max_tot_x_idx, 0]:.2f} at z = {z[max_tot_x_idx, 0]:.1f} nm")
    print(f"z-polarization: {tot[max_tot_z_idx, 0]:.2f} at z = {z[max_tot_z_idx, 0]:.1f} nm")
    
    # Computational efficiency note
    print(f"\nComputational efficiency:")
    print(f"Mirror symmetry reduces computation by using only 1/4 of sphere")
    print(f"This provides ~4x speedup compared to full sphere calculation")


def plot_comparison_details(tot, rad, tot0, rad0, z):
    """Detailed comparison plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total decay rates comparison
    axes[0, 0].plot(z.flatten(), tot[:, 0], 'r-', label='BEM', linewidth=2)
    axes[0, 0].plot(z.flatten(), tot0[:, 0], 'r--', label='Mie', linewidth=2)
    axes[0, 0].set_xlabel('Position (nm)')
    axes[0, 0].set_ylabel('Total decay rate')
    axes[0, 0].set_title('x-polarized dipole')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    axes[0, 1].plot(z.flatten(), tot[:, 1], 'b-', label='BEM', linewidth=2)
    axes[0, 1].plot(z.flatten(), tot0[:, 1], 'b--', label='Mie', linewidth=2)
    axes[0, 1].set_xlabel('Position (nm)')
    axes[0, 1].set_ylabel('Total decay rate')
    axes[0, 1].set_title('z-polarized dipole')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Relative differences
    rel_diff_tot_x = (tot[:, 0] - tot0[:, 0]) / tot0[:, 0] * 100
    rel_diff_tot_z = (tot[:, 1] - tot0[:, 1]) / tot0[:, 1] * 100
    
    axes[1, 0].plot(z.flatten(), rel_diff_tot_x, 'r-', linewidth=2)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Position (nm)')
    axes[1, 0].set_ylabel('Relative difference (%)')
    axes[1, 0].set_title('Total decay rate: (BEM - Mie)/Mie')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(z.flatten(), rel_diff_tot_z, 'b-', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Position (nm)')
    axes[1, 1].set_ylabel('Relative difference (%)')
    axes[1, 1].set_title('Total decay rate: (BEM - Mie)/Mie')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    tot, rad, tot0, rad0, z = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Gold nanosphere: {150} nm diameter")
    print(f"Dipole positions: z = {z[0, 0]:.0f} - {z[-1, 0]:.0f} nm")
    print(f"Number of positions: {len(z)}")
    print(f"Wavelength: 550 nm")
    print(f"Symmetry: xy mirror plane (1/4 sphere geometry)")
    print(f"Comparison: BEM with mirror symmetry vs analytical Mie theory")
    
    # Show detailed comparison
    plot_comparison_details(tot, rad, tot0, rad0, z)