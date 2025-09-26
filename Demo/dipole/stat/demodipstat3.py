"""
DEMODIPSTAT3 - Electric field for dipole above metallic nanosphere.

For a metallic nanosphere and an oscillating dipole located above the
sphere, with a dipole moment oriented along z, this program computes
the total electric field.

Runtime: ~3.8 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import trisphere, comparticle
from pymnpbem.misc import compoint, meshfield
from pymnpbem.bem import dipole, bemsolver


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
    
    # Compoint
    pt = compoint(p, [0, 0, 0.8 * diameter])
    
    # Dipole excitation (z-oriented)
    dip = dipole(pt, np.array([[0, 0, 1]]), op)
    
    print("Running quasistatic BEM simulation...")
    
    # BEM simulation
    # Set up BEM solver
    bem = bemsolver(p, op)
    
    # Surface charge
    sig = bem.solve(dip(p, enei))
    
    print("Computing electric field distribution...")
    
    # Computation of electric field
    # Mesh for calculation of electric field
    x_range = np.linspace(-15, 15, 81)
    z_range = np.linspace(-15, 15, 81)
    X, Z = np.meshgrid(x_range, z_range)
    
    # Object for electric field
    # MINDIST controls minimal distance to particle boundary
    emesh = meshfield(p, X, 0, Z, op, mindist=0.15)
    
    # Induced and incoming electric field
    e_induced = emesh(sig)
    e_incoming = emesh(dip.field(emesh.pt, enei))
    e_total = e_induced + e_incoming
    
    # Norm of electric field
    ee = np.sqrt(np.sum(np.abs(e_total)**2, axis=2))
    
    print("Creating visualizations...")
    
    # Final plot
    plt.figure(figsize=(12, 10))
    
    # Plot electric field (logarithmic scale)
    im = plt.imshow(np.log10(ee), extent=[X.min(), X.max(), Z.min(), Z.max()],
                   cmap='hot', aspect='equal', origin='lower')
    
    # Dipole position
    plt.plot(pt.pos[0], pt.pos[2], 'mo', markersize=12, 
            markerfacecolor='m', markeredgecolor='white', linewidth=2)
    
    # Color scale
    plt.clim([-3, 1])
    plt.colorbar(im, label='log₁₀(|E|)')
    
    plt.xlabel('x (nm)')
    plt.ylabel('z (nm)')
    plt.title('Electric field (logarithmic)')
    plt.tight_layout()
    plt.show()
    
    # Additional detailed analysis
    analyze_field_distribution(X, Z, e_total, ee, pt, diameter)
    
    return e_total, ee, X, Z, pt


def analyze_field_distribution(X, Z, e_total, ee, pt, diameter):
    """Analyze electric field distribution characteristics"""
    
    print("\n=== Field Distribution Analysis ===")
    
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
    
    # Field at dipole position (approximate)
    dipole_x_idx = np.argmin(np.abs(X[0, :] - pt.pos[0]))
    dipole_z_idx = np.argmin(np.abs(Z[:, 0] - pt.pos[2]))
    dipole_field = ee[dipole_z_idx, dipole_x_idx]
    
    print(f"Field at dipole position: {dipole_field:.2f}")
    
    # Field near sphere surface
    sphere_center = np.array([0, 0])  # x, z coordinates
    sphere_radius = diameter / 2
    
    # Points near sphere surface
    near_surface_mask = (np.sqrt(X**2 + Z**2) > sphere_radius - 1) & (np.sqrt(X**2 + Z**2) < sphere_radius + 1)
    if np.any(near_surface_mask):
        surface_field_avg = np.mean(ee[near_surface_mask])
        surface_field_max = np.max(ee[near_surface_mask])
        print(f"Average field near sphere surface: {surface_field_avg:.2f}")
        print(f"Maximum field near sphere surface: {surface_field_max:.2f}")
    
    # Field decay analysis along z-axis
    z_axis_idx = np.argmin(np.abs(X[0, :]))  # x = 0
    z_line = Z[:, z_axis_idx]
    field_z_line = ee[:, z_axis_idx]
    
    # Above sphere
    above_sphere_mask = z_line > sphere_radius
    if np.any(above_sphere_mask):
        z_above = z_line[above_sphere_mask]
        field_above = field_z_line[above_sphere_mask]
        print(f"Field decay along z-axis above sphere:")
        print(f"  At z = {z_above[0]:.1f} nm: |E| = {field_above[0]:.2f}")
        if len(z_above) > 10:
            print(f"  At z = {z_above[10]:.1f} nm: |E| = {field_above[10]:.2f}")


def plot_detailed_field_analysis(X, Z, e_total, ee, pt, diameter):
    """Detailed field component analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Total field magnitude
    im1 = axes[0, 0].imshow(np.log10(ee), extent=[X.min(), X.max(), Z.min(), Z.max()],
                           cmap='hot', aspect='equal', origin='lower')
    axes[0, 0].plot(pt.pos[0], pt.pos[2], 'mo', markersize=8)
    # Add circle for sphere boundary
    theta = np.linspace(0, 2*np.pi, 100)
    sphere_x = diameter/2 * np.cos(theta)
    sphere_z = diameter/2 * np.sin(theta)
    axes[0, 0].plot(sphere_x, sphere_z, 'w--', linewidth=1, alpha=0.7)
    axes[0, 0].set_title('Total field magnitude (log scale)')
    axes[0, 0].set_xlabel('x (nm)')
    axes[0, 0].set_ylabel('z (nm)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # z-component (dipole orientation)
    ez = np.abs(e_total[:, :, 2])
    im2 = axes[0, 1].imshow(np.log10(ez + 1e-10), extent=[X.min(), X.max(), Z.min(), Z.max()],
                           cmap='RdBu_r', aspect='equal', origin='lower')
    axes[0, 1].plot(pt.pos[0], pt.pos[2], 'ko', markersize=8)
    axes[0, 1].plot(sphere_x, sphere_z, 'k--', linewidth=1, alpha=0.7)
    axes[0, 1].set_title('|E_z| component (log scale)')
    axes[0, 1].set_xlabel('x (nm)')
    axes[0, 1].set_ylabel('z (nm)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # x-component
    ex = np.abs(e_total[:, :, 0])
    im3 = axes[1, 0].imshow(np.log10(ex + 1e-10), extent=[X.min(), X.max(), Z.min(), Z.max()],
                           cmap='viridis', aspect='equal', origin='lower')
    axes[1, 0].plot(pt.pos[0], pt.pos[2], 'wo', markersize=8)
    axes[1, 0].plot(sphere_x, sphere_z, 'w--', linewidth=1, alpha=0.7)
    axes[1, 0].set_title('|E_x| component (log scale)')
    axes[1, 0].set_xlabel('x (nm)')
    axes[1, 0].set_ylabel('z (nm)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Field along different lines
    # z-axis (x = 0)
    z_axis_idx = np.argmin(np.abs(X[0, :]))
    z_line = Z[:, z_axis_idx]
    field_z_line = ee[:, z_axis_idx]
    
    # x-axis at dipole height
    dipole_z_idx = np.argmin(np.abs(Z[:, 0] - pt.pos[2]))
    x_line = X[dipole_z_idx, :]
    field_x_line = ee[dipole_z_idx, :]
    
    axes[1, 1].semilogy(z_line, field_z_line, 'b-', label='Along z-axis (x=0)', linewidth=2)
    axes[1, 1].semilogy(x_line, field_x_line, 'r-', label=f'Along x-axis (z={pt.pos[2]:.1f})', linewidth=2)
    axes[1, 1].axvline(x=diameter/2, color='k', linestyle='--', alpha=0.5, label='Sphere edge')
    axes[1, 1].axvline(x=-diameter/2, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(x=pt.pos[2], color='m', linestyle=':', alpha=0.7, label='Dipole position')
    axes[1, 1].set_xlabel('Position (nm)')
    axes[1, 1].set_ylabel('|E|')
    axes[1, 1].set_title('Field profiles')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    e_total, ee, X, Z, pt = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Gold nanosphere: 10 nm diameter")
    print(f"Dipole: z-oriented at ({pt.pos[0]}, {pt.pos[1]}, {pt.pos[2]}) nm")
    print(f"Wavelength: 550 nm")
    print(f"Method: Quasistatic BEM")
    print(f"Field grid: {X.shape[0]} × {X.shape[1]} points")
    print(f"Field range: {X.min():.0f} to {X.max():.0f} nm")
    
    # Detailed analysis
    plot_detailed_field_analysis(X, Z, e_total, ee, pt, 10)