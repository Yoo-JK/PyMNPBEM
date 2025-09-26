"""
DEMODIPRET7 - Electric field for dipole above coated nanosphere.

For a coated metallic nanosphere (50 nm glass sphere coated with 10 nm
thick silver shell) and an oscillating dipole located above the
sphere, with a dipole moment oriented along z and the transition
energy tuned to the plasmon resonance, this program computes the
electric field map and the emission pattern using the full Maxwell
equations.

Runtime: ~17 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import trisphere, comparticle
from pymnpbem.misc import compoint, meshfield, spectrum, farfield
from pymnpbem.bem import dipole, bemsolver


def main():
    # Initialization
    # Options for BEM simulation
    op = bemoptions(sim='ret', interp='curv')
    
    # Table of dielectric functions
    epstab = [epsconst(1), epstable('gold.dat'), epsconst(2.25)]
    
    # Diameter of sphere
    diameter = 50
    
    # Nanospheres
    p1 = trisphere(144, diameter)        # Inner sphere (glass core)
    p2 = trisphere(256, diameter + 10)   # Outer sphere (gold shell)
    
    # Initialize sphere
    p = comparticle(epstab, [p1, p2], [[3, 2], [2, 1]], 1, 2, op)
    
    # Dipole oscillator
    # Dipole transition energy tuned to plasmon resonance frequency
    # Plasmon resonance extracted from DEMODIPRET6
    enei = 640
    
    # Compoint
    pt = compoint(p, [0, 0, 0.7 * diameter])
    
    # Dipole excitation (z-oriented)
    dip = dipole(pt, np.array([[0, 0, 1]]), op)
    
    print(f"Computing fields for dipole at plasmon resonance (λ = {enei} nm)...")
    
    # BEM simulation
    # Set up BEM solver
    bem = bemsolver(p, op)
    
    # Surface charge
    sig = bem.solve(dip(p, enei))
    
    print("Computing emission pattern...")
    
    # Emission pattern
    # Angles
    theta = np.linspace(0, 2 * np.pi, 101).reshape(-1, 1)
    
    # Directions for emission
    dir_emission = np.column_stack([np.cos(theta.flatten()), 
                                   np.zeros_like(theta.flatten()), 
                                   np.sin(theta.flatten())])
    
    # Set up spectrum object
    spec = spectrum(dir_emission, op)
    
    # Farfield radiation
    f = farfield(spec, sig)
    
    # Norm of Poynting vector
    poynting_cross = np.cross(f.e, np.conj(f.h), axis=1)
    s = np.linalg.norm(0.5 * np.real(poynting_cross), axis=1)
    
    print("Computing electric field map...")
    
    # Computation of electric field
    # Mesh for calculation of electric field
    x_range = np.linspace(-60, 60, 81)
    z_range = np.linspace(-60, 60, 81)
    X, Z = np.meshgrid(x_range, z_range)
    
    # Object for electric field
    # MINDIST controls the minimal distance of the field points to the particle boundary
    emesh = meshfield(p, X, 0, Z, op, mindist=0.5)
    
    # Induced and incoming electric field
    e_induced = emesh(sig)
    e_incoming = emesh(dip.field(emesh.pt, enei))
    e_total = e_induced + e_incoming
    
    # Norm of electric field
    ee = np.sqrt(np.sum(np.abs(e_total)**2, axis=2))
    
    print("Creating visualizations...")
    
    # Plot results
    plot_field_and_radiation(X, Z, ee, theta, s, pt, enei)
    plot_detailed_analysis(X, Z, e_total, theta, s, pt, enei)
    
    return e_total, ee, s, theta, X, Z, pt


def plot_field_and_radiation(X, Z, ee, theta, s, pt, wavelength):
    """Plot electric field map with radiation pattern overlay"""
    
    plt.figure(figsize=(12, 10))
    
    # Plot electric field (logarithmic scale)
    im = plt.imshow(np.log10(ee), extent=[X.min(), X.max(), Z.min(), Z.max()],
                   cmap='hot', aspect='equal', origin='lower')
    
    # Set color scale limits
    plt.clim([-4, -1])
    
    # Dipole position
    plt.plot(pt.pos[0], pt.pos[2], 'mo', markersize=10, 
            markerfacecolor='m', markeredgecolor='white', linewidth=2, 
            label='Dipole')
    
    # Cartesian coordinates of Poynting vector
    sx = 50 * s / np.max(s) * np.cos(theta.flatten())
    sy = 50 * s / np.max(s) * np.sin(theta.flatten())
    
    # Overlay with radiation pattern
    plt.plot(sx, sy, 'w--', linewidth=2, label='Radiation pattern')
    
    plt.colorbar(im, label='log₁₀(|E|)')
    plt.xlabel('x (nm)')
    plt.ylabel('z (nm)')
    plt.title(f'Electric field (log scale) and radiation pattern (λ = {wavelength} nm)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_detailed_analysis(X, Z, e_total, theta, s, pt, wavelength):
    """Detailed analysis plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Electric field components
    components = ['x', 'y', 'z']
    for i in range(3):
        if i < 2:
            row, col = 0, i
        else:
            row, col = 1, 0
            
        ee_comp = np.abs(e_total[:, :, i])
        im = axes[row, col].imshow(np.log10(ee_comp + 1e-10), 
                                  extent=[X.min(), X.max(), Z.min(), Z.max()],
                                  cmap='hot', aspect='equal', origin='lower')
        axes[row, col].set_title(f'|E_{components[i]}| (log scale)')
        axes[row, col].set_xlabel('x (nm)')
        axes[row, col].set_ylabel('z (nm)')
        plt.colorbar(im, ax=axes[row, col])
    
    # Radiation pattern (polar plot)
    ax_polar = plt.subplot(224, projection='polar')
    ax_polar.plot(theta.flatten(), s / np.max(s), 'r-', linewidth=2)
    ax_polar.set_title('Normalized radiation pattern')
    ax_polar.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()


def analyze_field_enhancement(X, Z, ee, pt):
    """Analyze field enhancement characteristics"""
    
    print("\n=== Field Enhancement Analysis ===")
    
    # Find maximum field enhancement
    max_idx = np.unravel_index(np.argmax(ee), ee.shape)
    max_field = ee[max_idx]
    max_x = X[max_idx]
    max_z = Z[max_idx]
    
    # Distance from dipole to maximum field point
    dist_to_max = np.sqrt((max_x - pt.pos[0])**2 + (max_z - pt.pos[2])**2)
    
    print(f"Maximum field enhancement: {max_field:.2f}")
    print(f"Location of maximum: ({max_x:.1f}, {max_z:.1f}) nm")
    print(f"Distance from dipole: {dist_to_max:.1f} nm")
    
    # Field enhancement at dipole position
    # Find closest grid point to dipole
    dipole_idx_x = np.argmin(np.abs(X[0, :] - pt.pos[0]))
    dipole_idx_z = np.argmin(np.abs(Z[:, 0] - pt.pos[2]))
    dipole_field = ee[dipole_idx_z, dipole_idx_x]
    
    print(f"Field at dipole position: {dipole_field:.2f}")
    
    # Average field in different regions
    center_mask = (X**2 + Z**2) < 25**2  # Within 25 nm of origin
    near_field_mask = (X**2 + Z**2) < 100**2  # Within 100 nm
    
    center_avg = np.mean(ee[center_mask])
    near_avg = np.mean(ee[near_field_mask])
    total_avg = np.mean(ee)
    
    print(f"Average field enhancement:")
    print(f"  Central region (r < 25 nm): {center_avg:.2f}")
    print(f"  Near field (r < 100 nm): {near_avg:.2f}")
    print(f"  Total region: {total_avg:.2f}")


def analyze_radiation_pattern(theta, s):
    """Analyze radiation pattern characteristics"""
    
    print("\n=== Radiation Pattern Analysis ===")
    
    # Normalize Poynting vector
    s_norm = s / np.max(s)
    
    # Find radiation maxima and minima
    max_indices = []
    min_indices = []
    
    for i in range(1, len(s_norm)-1):
        if s_norm[i] > s_norm[i-1] and s_norm[i] > s_norm[i+1] and s_norm[i] > 0.5:
            max_indices.append(i)
        elif s_norm[i] < s_norm[i-1] and s_norm[i] < s_norm[i+1] and s_norm[i] < 0.1:
            min_indices.append(i)
    
    print("Radiation maxima:")
    for idx in max_indices:
        angle_deg = np.degrees(theta[idx])
        print(f"  θ = {angle_deg[0]:.1f}°, Intensity = {s_norm[idx]:.3f}")
    
    print("Radiation minima:")
    for idx in min_indices:
        angle_deg = np.degrees(theta[idx])
        print(f"  θ = {angle_deg[0]:.1f}°, Intensity = {s_norm[idx]:.3f}")
    
    # Calculate directivity (simplified)
    total_power = np.trapz(s_norm, theta.flatten())
    max_power = np.max(s_norm)
    directivity = 2 * np.pi * max_power / total_power
    
    print(f"Approximate directivity: {directivity:.2f}")


if __name__ == '__main__':
    e_total, ee, s, theta, X, Z, pt = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Coated nanosphere: 50 nm glass core + 10 nm gold shell")
    print(f"Dipole: z-oriented at ({pt.pos[0]}, {pt.pos[1]}, {pt.pos[2]}) nm")
    print(f"Wavelength: 640 nm (plasmon resonance)")
    print(f"Field calculation grid: {X.shape[0]} × {X.shape[1]} points")
    print(f"Radiation pattern: {len(theta)} angular points")
    
    # Analyze results
    analyze_field_enhancement(X, Z, ee, pt)
    analyze_radiation_pattern(theta, s)