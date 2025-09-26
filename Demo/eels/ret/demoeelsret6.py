"""
DEMOEELSRET6 - Electric field map for EELS of nanotriangle.

For a silver nanotriangle with 80 nm base length and 10 nm height,
this program computes the electric field maps for (i) the excitation
of the dipole mode at 2.13 eV at the triangle edge and (ii) the
excitation of the breathing mode at 3.48 eV at the triangle center.

Runtime: ~51 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import polygon, tripolygon, comparticle
from pymnpbem.mesh2d import edgeprofile
from pymnpbem.misc import units, eelsbase, meshfield
from pymnpbem.bem import bemsolver, electronbeam


def main():
    # Initialization
    # Options for BEM simulation
    op = bemoptions(sim='ret', interp='curv')
    
    # Table of dielectric functions
    epstab = [epsconst(1), epstable('silver.dat')]
    
    # Dimensions of particle
    length = [80, 80 * 2 / np.sqrt(3)]
    
    # Polygon
    poly = polygon(3, size=length).round()
    
    # Edge profile
    edge = edgeprofile(10, 11)
    
    # Mesh data structure
    hdata = {'hmax': 5}
    
    # Extrude polygon
    p = tripolygon(poly, edge, hdata=hdata)
    
    # Make particle
    p = comparticle(epstab, [p], [2, 1], 1, op)
    
    # Width of electron beam and electron velocity
    width = 0.2
    vel = eelsbase.ene2vel(200e3)  # 200 keV electron energy
    
    # Impact parameters (triangle edge and center)
    imp = np.array([[-40, 0], [0, 0]])  # Edge, Center
    
    # Loss energies in eV (dipole and breathing mode)
    ene = np.array([2.13, 3.58])
    
    # Convert energies to nm
    unit_converter = units()
    enei = unit_converter.eV2nm / ene
    
    print("Running BEM simulation for electric field maps...")
    
    # BEM simulation
    # BEM solver
    bem = bemsolver(p, op)
    
    # Electron beam excitation
    exc1 = electronbeam(p, imp[0, :], width, vel, op)  # Edge excitation
    exc2 = electronbeam(p, imp[1, :], width, vel, op)  # Center excitation
    
    # Surface charges
    sig1 = bem.solve(exc1(enei[0]))  # Dipole mode
    sig2 = bem.solve(exc2(enei[1]))  # Breathing mode
    
    print("Computing electric field distributions...")
    
    # Computation of electric field
    # Mesh for calculation of electric field
    x_range = np.linspace(-60, 60, 81)
    z_range = np.linspace(-60, 60, 81)
    X, Z = np.meshgrid(x_range, z_range)
    
    # Object for electric field
    # MINDIST controls minimal distance to particle boundary
    emesh = meshfield(p, X, 0, Z, op, mindist=0.4, nmax=3000)
    
    # Induced electric field
    e1 = emesh(sig1)  # Dipole mode field
    e2 = emesh(sig2)  # Breathing mode field
    
    # Field magnitudes
    ee1 = np.sqrt(np.sum(np.abs(e1)**2, axis=2))
    ee2 = np.sqrt(np.sum(np.abs(e2)**2, axis=2))
    
    print("Creating field visualizations...")
    
    # Final plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot electric field for dipole mode
    im1 = axes[0].imshow(np.log10(ee1), extent=[X.min(), X.max(), Z.min(), Z.max()],
                        cmap='hot', aspect='equal', origin='lower')
    
    # Plot electron trajectory
    axes[0].plot([imp[0, 0], imp[0, 0]], [-60, 60], 'w:', linewidth=2, alpha=0.8)
    axes[0].set_xlabel('x (nm)')
    axes[0].set_ylabel('z (nm)')
    axes[0].set_title('Dipole mode (log. scale)')
    
    # Colorbar
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('log₁₀(|E|)')
    im1.set_clim([-1, 1])
    
    # Plot electric field for breathing mode
    im2 = axes[1].imshow(np.log10(ee2), extent=[X.min(), X.max(), Z.min(), Z.max()],
                        cmap='hot', aspect='equal', origin='lower')
    
    # Plot electron trajectory
    axes[1].plot([imp[1, 0], imp[1, 0]], [-60, 60], 'w:', linewidth=2, alpha=0.8)
    axes[1].set_xlabel('x (nm)')
    axes[1].set_ylabel('z (nm)')
    axes[1].set_title('Breathing mode (log. scale)')
    
    # Colorbar
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('log₁₀(|E|)')
    im2.set_clim([-1, 1])
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    analyze_mode_characteristics(X, Z, e1, e2, ee1, ee2, imp, ene)
    
    return e1, e2, ee1, ee2, X, Z, imp, ene


def analyze_mode_characteristics(X, Z, e1, e2, ee1, ee2, imp, ene):
    """Analyze characteristics of dipole and breathing modes"""
    
    print("\n=== Mode Characteristics Analysis ===")
    
    mode_names = ['Dipole mode (edge excitation)', 'Breathing mode (center excitation)']
    field_data = [ee1, ee2]
    energies = ene
    impact_params = imp
    
    for i, (mode_name, field, energy, impact) in enumerate(zip(mode_names, field_data, energies, impact_params)):
        print(f"\n{mode_name} at {energy:.2f} eV:")
        
        # Maximum field enhancement
        max_field = np.max(field)
        max_idx = np.unravel_index(np.argmax(field), field.shape)
        max_x = X[max_idx]
        max_z = Z[max_idx]
        
        print(f"  Maximum field: {max_field:.2f}")
        print(f"  Maximum position: ({max_x:.1f}, {max_z:.1f}) nm")
        print(f"  Impact parameter: ({impact[0]:.1f}, {impact[1]:.1f}) nm")
        
        # Field distribution characteristics
        # Calculate moments to characterize field distribution
        total_intensity = np.sum(field)
        if total_intensity > 0:
            com_x = np.sum(X * field) / total_intensity
            com_z = np.sum(Z * field) / total_intensity
            print(f"  Center of mass: ({com_x:.1f}, {com_z:.1f}) nm")
        
        # Spatial extent (FWHM-like measure)
        max_half = max_field / 2
        extent_mask = field > max_half
        if np.any(extent_mask):
            x_extent = np.max(X[extent_mask]) - np.min(X[extent_mask])
            z_extent = np.max(Z[extent_mask]) - np.min(Z[extent_mask])
            print(f"  Spatial extent (>50% max): x = {x_extent:.1f} nm, z = {z_extent:.1f} nm")


def plot_detailed_field_analysis(X, Z, e1, e2, ee1, ee2, imp):
    """Detailed field analysis plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Field magnitude maps
    im1 = axes[0, 0].imshow(np.log10(ee1), extent=[X.min(), X.max(), Z.min(), Z.max()],
                           cmap='hot', aspect='equal', origin='lower')
    axes[0, 0].plot([imp[0, 0], imp[0, 0]], [-60, 60], 'w:', linewidth=2)
    axes[0, 0].set_title('Dipole mode: |E|')
    axes[0, 0].set_xlabel('x (nm)')
    axes[0, 0].set_ylabel('z (nm)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[1, 0].imshow(np.log10(ee2), extent=[X.min(), X.max(), Z.min(), Z.max()],
                           cmap='hot', aspect='equal', origin='lower')
    axes[1, 0].plot([imp[1, 0], imp[1, 0]], [-60, 60], 'w:', linewidth=2)
    axes[1, 0].set_title('Breathing mode: |E|')
    axes[1, 0].set_xlabel('x (nm)')
    axes[1, 0].set_ylabel('z (nm)')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # x-component of electric field
    ex1 = np.abs(e1[:, :, 0])
    ex2 = np.abs(e2[:, :, 0])
    
    im3 = axes[0, 1].imshow(np.log10(ex1 + 1e-10), extent=[X.min(), X.max(), Z.min(), Z.max()],
                           cmap='RdBu_r', aspect='equal', origin='lower')
    axes[0, 1].set_title('Dipole mode: |E_x|')
    axes[0, 1].set_xlabel('x (nm)')
    axes[0, 1].set_ylabel('z (nm)')
    plt.colorbar(im3, ax=axes[0, 1])
    
    im4 = axes[1, 1].imshow(np.log10(ex2 + 1e-10), extent=[X.min(), X.max(), Z.min(), Z.max()],
                           cmap='RdBu_r', aspect='equal', origin='lower')
    axes[1, 1].set_title('Breathing mode: |E_x|')
    axes[1, 1].set_xlabel('x (nm)')
    axes[1, 1].set_ylabel('z (nm)')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # z-component of electric field
    ez1 = np.abs(e1[:, :, 2])
    ez2 = np.abs(e2[:, :, 2])
    
    im5 = axes[0, 2].imshow(np.log10(ez1 + 1e-10), extent=[X.min(), X.max(), Z.min(), Z.max()],
                           cmap='viridis', aspect='equal', origin='lower')
    axes[0, 2].set_title('Dipole mode: |E_z|')
    axes[0, 2].set_xlabel('x (nm)')
    axes[0, 2].set_ylabel('z (nm)')
    plt.colorbar(im5, ax=axes[0, 2])
    
    im6 = axes[1, 2].imshow(np.log10(ez2 + 1e-10), extent=[X.min(), X.max(), Z.min(), Z.max()],
                           cmap='viridis', aspect='equal', origin='lower')
    axes[1, 2].set_title('Breathing mode: |E_z|')
    axes[1, 2].set_xlabel('x (nm)')
    axes[1, 2].set_ylabel('z (nm)')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.show()


def plot_field_line_profiles(X, Z, ee1, ee2, imp):
    """Plot field profiles along specific lines"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Profile along x-axis (z=0)
    z_center_idx = np.argmin(np.abs(Z[:, 0]))
    x_line = X[z_center_idx, :]
    
    dipole_profile_x = ee1[z_center_idx, :]
    breathing_profile_x = ee2[z_center_idx, :]
    
    axes[0, 0].plot(x_line, dipole_profile_x, 'r-', label='Dipole mode', linewidth=2)
    axes[0, 0].plot(x_line, breathing_profile_x, 'b-', label='Breathing mode', linewidth=2)
    axes[0, 0].axvline(x=imp[0, 0], color='r', linestyle='--', alpha=0.7, label='Dipole impact')
    axes[0, 0].axvline(x=imp[1, 0], color='b', linestyle='--', alpha=0.7, label='Breathing impact')
    axes[0, 0].set_xlabel('x (nm)')
    axes[0, 0].set_ylabel('|E|')
    axes[0, 0].set_title('Field profile along x-axis (z=0)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Profile along z-axis (x=0)
    x_center_idx = np.argmin(np.abs(X[0, :]))
    z_line = Z[:, x_center_idx]
    
    dipole_profile_z = ee1[:, x_center_idx]
    breathing_profile_z = ee2[:, x_center_idx]
    
    axes[0, 1].plot(z_line, dipole_profile_z, 'r-', label='Dipole mode', linewidth=2)
    axes[0, 1].plot(z_line, breathing_profile_z, 'b-', label='Breathing mode', linewidth=2)
    axes[0, 1].set_xlabel('z (nm)')
    axes[0, 1].set_ylabel('|E|')
    axes[0, 1].set_title('Field profile along z-axis (x=0)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Radial profiles from triangle center
    center = [0, 0]
    distances = np.sqrt((X - center[0])**2 + (Z - center[1])**2)
    
    # Create radial bins
    r_bins = np.linspace(0, 60, 30)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    
    dipole_radial = np.zeros_like(r_centers)
    breathing_radial = np.zeros_like(r_centers)
    
    for i in range(len(r_centers)):
        mask = (distances >= r_bins[i]) & (distances < r_bins[i+1])
        if np.any(mask):
            dipole_radial[i] = np.mean(ee1[mask])
            breathing_radial[i] = np.mean(ee2[mask])
    
    axes[1, 0].plot(r_centers, dipole_radial, 'r-o', label='Dipole mode', linewidth=2)
    axes[1, 0].plot(r_centers, breathing_radial, 'b-s', label='Breathing mode', linewidth=2)
    axes[1, 0].set_xlabel('Radial distance (nm)')
    axes[1, 0].set_ylabel('Average |E|')
    axes[1, 0].set_title('Radial field profiles')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Field ratio comparison
    field_ratio = ee1 / (ee2 + 1e-10)
    
    im = axes[1, 1].imshow(np.log10(field_ratio), extent=[X.min(), X.max(), Z.min(), Z.max()],
                          cmap='RdBu_r', aspect='equal', origin='lower')
    axes[1, 1].set_xlabel('x (nm)')
    axes[1, 1].set_ylabel('z (nm)')
    axes[1, 1].set_title('Field ratio: Dipole/Breathing')
    plt.colorbar(im, ax=axes[1, 1], label='log₁₀(|E₁|/|E₂|)')
    
    plt.tight_layout()
    plt.show()


def explain_triangle_modes():
    """Explain triangle plasmon modes"""
    
    print("\n=== Triangle Plasmon Modes ===")
    print("Dipole mode (2.13 eV):")
    print("- Excited preferentially at triangle edges")
    print("- Shows asymmetric field distribution")
    print("- Strong field enhancement at corners")
    print("- Lower energy fundamental mode")
    
    print("\nBreathing mode (3.58 eV):")
    print("- Excited preferentially at triangle center")
    print("- Shows more symmetric field distribution")
    print("- Uniform expansion/contraction of electron cloud")
    print("- Higher energy mode with radial symmetry")


if __name__ == '__main__':
    e1, e2, ee1, ee2, X, Z, imp, ene = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Silver nanotriangle: 80 nm base length, 10 nm height")
    print(f"Electron beam: 200 keV energy, 0.2 nm width")
    print(f"Mode 1: Dipole mode at {ene[0]:.2f} eV, impact at ({imp[0,0]:.0f}, {imp[0,1]:.0f}) nm")
    print(f"Mode 2: Breathing mode at {ene[1]:.2f} eV, impact at ({imp[1,0]:.0f}, {imp[1,1]:.0f}) nm")
    print(f"Field calculation: {X.shape[0]} × {X.shape[1]} grid")
    
    # Detailed analysis
    plot_detailed_field_analysis(X, Z, e1, e2, ee1, ee2, imp)
    plot_field_line_profiles(X, Z, ee1, ee2, imp)
    explain_triangle_modes()