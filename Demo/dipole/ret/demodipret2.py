"""
DEMODIPRET2 - Energy dependent lifetime for dipole above a nanosphere.

For a metallic nanosphere and an oscillating dipole located above the
sphere, with a dipole moment oriented along x or z, this program
computes the total and radiative dipole scattering rates for different
dipole transition energies using the full Maxwell equations, and
compares the results with Mie theory.

Runtime: ~60 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import trispheresegment, comparticle
from pymnpbem.misc import compoint
from pymnpbem.bem import dipole, bemsolver
from pymnpbem.mie import miesolver


def main():
    # Initialization
    # Options for BEM simulation
    op = bemoptions(sim='ret', interp='curv')
    
    # Table of dielectric functions
    epstab = [epsconst(1), epstable('gold.dat')]
    
    # Diameter of sphere
    diameter = 150
    
    # Nanosphere with finer discretization at the top
    phi_grid = 2 * np.pi * np.linspace(0, 1, 31)
    theta_grid = np.pi * np.linspace(0, 1, 31) ** 2
    p_mesh = trispheresegment(phi_grid, theta_grid, diameter)
    
    # Initialize sphere
    p = comparticle(epstab, [p_mesh], [2, 1], 1, op)
    
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
    
    print("Running BEM simulation...")
    
    # Loop over wavelengths with progress bar
    for ien in tqdm(range(len(enei)), desc="BEM solver", ncols=80):
        # Surface charge
        sig = bem.solve(dip(p, enei[ien]))  # Using solve method instead of \ operator
        
        # Total and radiative decay rate
        tot[ien, :], rad[ien, :] = dip.decayrate(sig)
    
    print("BEM simulation completed!")
    
    # Final plot
    plt.figure(figsize=(12, 8))
    
    # Plot BEM results
    plt.plot(enei, tot[:, 0], '-', label='tot(x) @ BEM', linewidth=2)
    plt.plot(enei, tot[:, 1], '-', label='tot(z) @ BEM', linewidth=2)
    plt.plot(enei, rad[:, 0], 'o-', label='rad(x) @ BEM', markersize=4)
    plt.plot(enei, rad[:, 1], 'o-', label='rad(z) @ BEM', markersize=4)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Total decay rate')
    plt.grid(True, alpha=0.3)
    
    # Comparison with Mie theory
    print("Running Mie theory calculation...")
    mie = miesolver(epstab[1], epstab[0], diameter, op)
    
    # Total and radiative decay rate for Mie theory
    tot0 = np.zeros((len(enei), 2))
    rad0 = np.zeros((len(enei), 2))
    
    # Loop over energies
    for ien in tqdm(range(len(enei)), desc="Mie theory", ncols=80):
        tot0[ien, :], rad0[ien, :] = mie.decayrate(enei[ien], pt.pos[:, 2])  # z-coordinate
    
    # Plot Mie theory results
    plt.plot(enei, tot0[:, 0], '--', label='tot(x) @ Mie', linewidth=2)
    plt.plot(enei, tot0[:, 1], '--', label='tot(z) @ Mie', linewidth=2)
    plt.plot(enei, rad0[:, 0], 'o--', label='rad(x) @ Mie', markersize=4)
    plt.plot(enei, rad0[:, 1], 'o--', label='rad(z) @ Mie', markersize=4)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'Energy dependent decay rates for dipole at z = {0.7 * diameter:.1f} nm')
    plt.tight_layout()
    plt.show()
    
    print("Mie theory calculation completed!")
    
    return tot, rad, tot0, rad0, enei


def compare_results(tot, rad, tot0, rad0, enei):
    """Compare BEM and Mie theory results"""
    
    # Calculate relative differences
    rel_diff_tot_x = np.abs(tot[:, 0] - tot0[:, 0]) / tot0[:, 0] * 100
    rel_diff_tot_z = np.abs(tot[:, 1] - tot0[:, 1]) / tot0[:, 1] * 100
    rel_diff_rad_x = np.abs(rad[:, 0] - rad0[:, 0]) / rad0[:, 0] * 100
    rel_diff_rad_z = np.abs(rad[:, 1] - rad0[:, 1]) / rad0[:, 1] * 100
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(enei, rel_diff_tot_x, 'r-', label='Total (x)')
    plt.plot(enei, rel_diff_tot_z, 'b-', label='Total (z)')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Relative difference (%)')
    plt.title('Total decay rate: BEM vs Mie')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(enei, rel_diff_rad_x, 'r-', label='Radiative (x)')
    plt.plot(enei, rel_diff_rad_z, 'b-', label='Radiative (z)')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Relative difference (%)')
    plt.title('Radiative decay rate: BEM vs Mie')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n=== Comparison Statistics ===")
    print(f"Total decay rate (x) - Mean rel. diff: {np.mean(rel_diff_tot_x):.2f}%, Max: {np.max(rel_diff_tot_x):.2f}%")
    print(f"Total decay rate (z) - Mean rel. diff: {np.mean(rel_diff_tot_z):.2f}%, Max: {np.max(rel_diff_tot_z):.2f}%")
    print(f"Radiative decay rate (x) - Mean rel. diff: {np.mean(rel_diff_rad_x):.2f}%, Max: {np.max(rel_diff_rad_x):.2f}%")
    print(f"Radiative decay rate (z) - Mean rel. diff: {np.mean(rel_diff_rad_z):.2f}%, Max: {np.max(rel_diff_rad_z):.2f}%")


if __name__ == '__main__':
    tot, rad, tot0, rad0, enei = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Energy range: {enei[0]:.0f} - {enei[-1]:.0f} nm")
    print(f"Number of energy points: {len(enei)}")
    print(f"Sphere diameter: 150 nm")
    print(f"Dipole position: z = {0.7 * 150:.1f} nm")
    
    # Show detailed comparison
    compare_results(tot, rad, tot0, rad0, enei)