"""
DEMODIPSTAT9 - Dipole lifetime for nanosphere and mirror symmetry.

For a metallic nanosphere and an oscillating dipole located above the
sphere, with a dipole moment oriented along x or z, this program
computes the total and radiative dipole scattering rates using mirror
symmetry within the quasistatic approximation, and compares the
results with Mie theory.

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
    # Options for BEM simulation (quasistatic with mirror symmetry)
    op = bemoptions(sim='stat', waitbar=0, interp='curv', sym='xy')
    
    # Table of dielectric functions
    epstab = [epsconst(4), epstable('gold.dat')]  # Note: different order than previous examples
    
    # Diameter of sphere
    diameter = 10
    
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
    
    print("Running quasistatic BEM simulation with mirror symmetry...")
    
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
    analyze_quasistatic_mirror_symmetry(tot, rad, tot0, rad0, z)
    
    return tot, rad, tot0, rad0, z


def analyze_quasistatic_mirror_symmetry(tot, rad, tot0, rad0, z):
    """Analyze benefits of mirror symmetry in quasistatic calculations"""
    
    print("\n=== Quasistatic Mirror Symmetry Analysis ===")
    
    # Calculate relative differences between quasistatic BEM and Mie
    rel_diff_tot_x = np.abs(tot[:, 0] - tot0[:, 0]) / (tot0[:, 0] + 1e-10) * 100
    rel_diff_tot_z = np.abs(tot[:, 1] - tot0[:, 1]) / (tot0[:, 1] + 1e-10) * 100
    rel_diff_rad_x = np.abs(rad[:, 0] - rad0[:, 0]) / (rad0[:, 0] + 1e-10) * 100
    rel_diff_rad_z = np.abs(rad[:, 1] - rad0[:, 1]) / (rad0[:, 1] + 1e-10) * 100
    
    print("Quasistatic BEM vs Mie theory agreement:")
    print(f"Total decay rate (x): Mean diff = {np.mean(rel_diff_tot_x):.2f}%, Max diff = {np.max(rel_diff_tot_x):.2f}%")
    print(f"Total decay rate (z): Mean diff = {np.mean(rel_diff_tot_z):.2f}%, Max diff = {np.max(rel_diff_tot_z):.2f}%")
    print(f"Radiative decay rate (x): Mean diff = {np.mean(rel_diff_rad_x):.2f}%, Max diff = {np.max(rel_diff_rad_x):.2f}%")
    print(f"Radiative decay rate (z): Mean diff = {np.mean(rel_diff_rad_z):.2f}%, Max diff = {np.max(rel_diff_rad_z):.2f}%")
    
    # Find positions of maximum enhancement
    max_tot_x_idx = np.argmax(tot[:, 0])
    max_tot_z_idx = np.argmax(tot[:, 1])
    
    print(f"\nMaximum enhancements (quasistatic with mirror symmetry):")
    print(f"x-polarization: {tot[max_tot_x_idx, 0]:.2f} at z = {z[max_tot_x_idx, 0]:.1f} nm")
    print(f"z-polarization: {tot[max_tot_z_idx, 0]:.2f} at z = {z[max_tot_z_idx, 0]:.1f} nm")
    
    # Computational efficiency
    print(f"\nComputational efficiency:")
    print(f"Mirror symmetry (xy plane) reduces computation by 4x")
    print(f"Quasistatic approximation eliminates retardation integrals")
    print(f"Combined: Very fast computation with good accuracy for small particles")
    
    # Validity analysis
    diameter = 10  # nm
    wavelength = 550  # nm
    size_parameter = np.pi * diameter / wavelength
    
    print(f"\nApproximation validity:")
    print(f"Particle diameter: {diameter} nm")
    print(f"Wavelength: {wavelength} nm")
    print(f"Size parameter (πD/λ): {size_parameter:.3f}")
    
    if size_parameter < 0.2:
        print("Quasistatic approximation is excellent for this size")
    elif size_parameter < 0.5:
        print("Quasistatic approximation is good for this size")
    else:
        print("Quasistatic approximation may have limitations for this size")


def plot_comparison_details(tot, rad, tot0, rad0, z):
    """Detailed comparison plots for quasistatic vs Mie"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total decay rates comparison
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
    
    # Relative differences
    rel_diff_tot_x = (tot[:, 0] - tot0[:, 0]) / (tot0[:, 0] + 1e-10) * 100
    rel_diff_tot_z = (tot[:, 1] - tot0[:, 1]) / (tot0[:, 1] + 1e-10) * 100
    
    axes[1, 0].plot(z.flatten(), rel_diff_tot_x, 'r-', linewidth=2)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Position (nm)')
    axes[1, 0].set_ylabel('Relative difference (%)')
    axes[1, 0].set_title('Total decay rate: (Quasistatic - Mie)/Mie (x-pol)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(z.flatten(), rel_diff_tot_z, 'b-', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Position (nm)')
    axes[1, 1].set_ylabel('Relative difference (%)')
    axes[1, 1].set_title('Total decay rate: (Quasistatic - Mie)/Mie (z-pol)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_quantum_efficiency(tot, rad, tot0, rad0, z):
    """Analyze quantum efficiency in quasistatic vs full theory"""
    
    print("\n=== Quantum Efficiency Analysis ===")
    
    # Calculate quantum efficiencies
    qe_qs = rad / (tot + 1e-10)
    qe_mie = rad0 / (tot0 + 1e-10)
    
    # Average quantum efficiencies
    avg_qe_qs_x = np.mean(qe_qs[:, 0])
    avg_qe_qs_z = np.mean(qe_qs[:, 1])
    avg_qe_mie_x = np.mean(qe_mie[:, 0])
    avg_qe_mie_z = np.mean(qe_mie[:, 1])
    
    print(f"Average quantum efficiencies:")
    print(f"  Quasistatic - x-pol: {avg_qe_qs_x:.3f}, z-pol: {avg_qe_qs_z:.3f}")
    print(f"  Mie theory - x-pol: {avg_qe_mie_x:.3f}, z-pol: {avg_qe_mie_z:.3f}")
    
    # Plot quantum efficiency comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(z.flatten(), qe_qs[:, 0], 'r-', label='Quasistatic (x)', linewidth=2)
    plt.plot(z.flatten(), qe_qs[:, 1], 'b-', label='Quasistatic (z)', linewidth=2)
    plt.plot(z.flatten(), qe_mie[:, 0], 'r--', label='Mie (x)', linewidth=2)
    plt.plot(z.flatten(), qe_mie[:, 1], 'b--', label='Mie (z)', linewidth=2)
    plt.xlabel('Position (nm)')
    plt.ylabel('Quantum efficiency')
    plt.title('Quantum efficiency comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    # Non-radiative contributions
    nonrad_qs = tot - rad
    nonrad_mie = tot0 - rad0
    
    plt.subplot(1, 2, 2)
    plt.semilogy(z.flatten(), nonrad_qs[:, 0], 'r-', label='Quasistatic (x)', linewidth=2)
    plt.semilogy(z.flatten(), nonrad_qs[:, 1], 'b-', label='Quasistatic (z)', linewidth=2)
    plt.semilogy(z.flatten(), nonrad_mie[:, 0], 'r--', label='Mie (x)', linewidth=2)
    plt.semilogy(z.flatten(), nonrad_mie[:, 1], 'b--', label='Mie (z)', linewidth=2)
    plt.xlabel('Position (nm)')
    plt.ylabel('Non-radiative decay rate')
    plt.title('Non-radiative decay comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    tot, rad, tot0, rad0, z = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Gold nanosphere: 10 nm diameter")
    print(f"Dipole positions: z = {z[0, 0]:.1f} - {z[-1, 0]:.1f} nm")
    print(f"Number of positions: {len(z)}")
    print(f"Wavelength: 550 nm")
    print(f"Method: Quasistatic BEM with xy mirror symmetry")
    print(f"Background medium: ε = 4 (instead of air)")
    print(f"Computational speedup: ~4x from mirror symmetry + quasistatic")
    
    # Detailed analysis
    plot_comparison_details(tot, rad, tot0, rad0, z)
    analyze_quantum_efficiency(tot, rad, tot0, rad0, z)