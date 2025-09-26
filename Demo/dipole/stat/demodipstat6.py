"""
DEMODIPSTAT6 - Electric field for dipole between sphere and layer.

For a metallic nanosphere with a diameter of 4 nm located 1 nm above a
substrate and a dipole located between sphere and layer, this program
computes the radiation pattern and the total electric field using the
quasistatic approximation.

Runtime: ~22 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import trispheresegment, comparticle, shift, flip
from pymnpbem.misc import compoint, layerstructure, meshfield, spectrum, farfield
from pymnpbem.bem import dipole, bemsolver


def main():
    # Initialization
    # Table of dielectric functions
    epstab = [epsconst(1), epstable('gold.dat'), epsconst(2.25)]  # air, gold, substrate
    
    # Location of interface of substrate
    ztab = 0
    
    # Default options for layer structure
    opt = layerstructure.options()
    
    # Set up layer structure
    layer = layerstructure(epstab, [1, 3], ztab, opt)  # air above, substrate below
    
    # Options for BEM simulation (quasistatic with layer)
    op = bemoptions(sim='stat', interp='curv', layer=layer)
    
    # Nanosphere with finer discretization at the bottom
    phi_grid = 2 * np.pi * np.linspace(0, 1, 31)
    theta_grid = np.pi * np.linspace(0, 1, 31) ** 2
    p_sphere = trispheresegment(phi_grid, theta_grid, 4)  # 4 nm diameter
    
    # Flip sphere (finer discretization at bottom)
    p_sphere = flip(p_sphere, axis=2)
    
    # Place nanosphere 1 nm above substrate
    min_z = np.min(p_sphere.pos[:, 2])
    p_sphere = shift(p_sphere, [0, 0, -min_z + 1 + ztab])
    
    # Set up COMPARTICLE object
    p = comparticle(epstab, [p_sphere], [2, 1], 1, op)
    
    # Dipole oscillator
    enei = 550
    
    # Compoint
    pt = compoint(p, [1, 0, 0.5], op)  # Between sphere and substrate
    
    # Dipole excitation (z-oriented)
    dip = dipole(pt, np.array([[0, 0, 1]]), op)
    
    print("Running quasistatic BEM simulation...")
    
    # BEM simulation
    # Set up BEM solver
    bem = bemsolver(p, op)
    
    # Surface charge
    sig = bem.solve(dip(p, enei))
    
    print("Computing emission pattern...")
    
    # Emission pattern
    # Angles
    theta = np.linspace(0, 2 * np.pi, 301).reshape(-1, 1)
    
    # Directions for emission
    dir_emission = np.column_stack([np.cos(theta.flatten()), 
                                   np.zeros_like(theta.flatten()), 
                                   np.sin(theta.flatten())])
    
    # Set up spectrum object
    spec = spectrum(dir_emission, op)
    
    # Farfield radiation (scattered + dipole radiation)
    f_scattered = farfield(spec, sig)
    f_dipole = farfield(dip, spec, enei)
    f_total = f_scattered + f_dipole
    
    # Norm of Poynting vector
    poynting_cross = np.cross(f_total.e, np.conj(f_total.h), axis=1)
    s = np.linalg.norm(0.5 * np.real(poynting_cross), axis=1)
    
    print("Computing electric field distribution...")
    
    # Computation of electric field
    # Mesh for calculation of electric field
    x_range = np.linspace(-8, 8, 81)
    z_range = np.linspace(-8, 8, 81)
    X, Z = np.meshgrid(x_range, z_range)
    
    # Object for electric field calculation
    emesh = meshfield(p, X, 0, Z, op, mindist=0.15, nmax=3000)
    
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
    
    # Plot substrate interface
    plt.plot([X.min(), X.max()], [ztab, ztab], 'w--', linewidth=2)
    
    # Dipole position
    plt.plot(pt.pos[0], pt.pos[2], 'mo', markersize=12, 
            markerfacecolor='m', markeredgecolor='white', linewidth=2)
    
    # Cartesian coordinates of Poynting vector (radiation pattern)
    sx = 8 * s / np.max(s) * np.cos(theta.flatten())
    sy = 8 * s / np.max(s) * np.sin(theta.flatten())
    
    # Overlay radiation pattern
    plt.plot(sx, sy, 'w-', linewidth=2)
    
    # Color scale
    plt.clim([-2, 2])
    plt.colorbar(im, label='log₁₀(|E|)')
    
    plt.xlabel('x (nm)')
    plt.ylabel('z (nm)')
    plt.title('Electric field (logarithmic), radiation pattern (quasistatic)')
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    analyze_quasistatic_fields(X, Z, e_total, ee, theta, s, pt, ztab)
    
    return e_total, ee, s, theta, X, Z, pt


def analyze_quasistatic_fields(X, Z, e_total, ee, theta, s, pt, ztab):
    """Analyze quasistatic field characteristics"""
    
    print("\n=== Quasistatic Field Analysis ===")
    
    # Find maximum field enhancement
    max_idx = np.unravel_index(np.argmax(ee), ee.shape)
    max_field = ee[max_idx]
    max_x = X[max_idx]
    max_z = Z[max_idx]
    
    print(f"Maximum field enhancement: {max_field:.2f}")
    print(f"Location: ({max_x:.1f}, {max_z:.1f}) nm")
    print(f"Distance from dipole: {np.sqrt((max_x-pt.pos[0])**2 + (max_z-pt.pos[2])**2):.1f} nm")
    
    # Field at dipole position
    dipole_x_idx = np.argmin(np.abs(X[0, :] - pt.pos[0]))
    dipole_z_idx = np.argmin(np.abs(Z[:, 0] - pt.pos[2]))
    local_field = ee[dipole_z_idx, dipole_x_idx]
    
    print(f"Local field at dipole: {local_field:.2f}")
    
    # Substrate effects
    above_mask = Z > ztab + 0.5
    below_mask = Z < ztab - 0.5
    
    if np.any(above_mask) and np.any(below_mask):
        above_avg = np.mean(ee[above_mask])
        below_avg = np.mean(ee[below_mask])
        print(f"Field ratio (above/below substrate): {above_avg/below_avg:.2f}")
    
    # Radiation pattern analysis
    print(f"\nRadiation pattern (quasistatic approximation):")
    s_norm = s / np.max(s)
    
    # Find main radiation directions
    peak_indices = []
    for i in range(1, len(s_norm)-1):
        if s_norm[i] > s_norm[i-1] and s_norm[i] > s_norm[i+1] and s_norm[i] > 0.7:
            peak_indices.append(i)
    
    if peak_indices:
        for i, idx in enumerate(peak_indices[:3]):  # Show up to 3 peaks
            angle_deg = np.degrees(theta[idx, 0])
            print(f"  Peak {i+1}: θ = {angle_deg:.1f}°, Intensity = {s_norm[idx]:.3f}")
    else:
        max_idx = np.argmax(s_norm)
        angle_deg = np.degrees(theta[max_idx, 0])
        print(f"  Main direction: θ = {angle_deg:.1f}°, Intensity = {s_norm[max_idx]:.3f}")


def plot_detailed_quasistatic_analysis(X, Z, e_total, ee, theta, s, pt, ztab):
    """Detailed analysis plots for quasistatic fields"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Total field with substrate interface
    im1 = axes[0, 0].imshow(np.log10(ee), extent=[X.min(), X.max(), Z.min(), Z.max()],
                           cmap='hot', aspect='equal', origin='lower')
    axes[0, 0].plot([X.min(), X.max()], [ztab, ztab], 'w--', linewidth=1)
    axes[0, 0].plot(pt.pos[0], pt.pos[2], 'mo', markersize=8)
    axes[0, 0].set_title('Electric field (log scale)')
    axes[0, 0].set_xlabel('x (nm)')
    axes[0, 0].set_ylabel('z (nm)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # z-component (dipole orientation)
    ez = np.abs(e_total[:, :, 2])
    im2 = axes[0, 1].imshow(np.log10(ez + 1e-10), extent=[X.min(), X.max(), Z.min(), Z.max()],
                           cmap='RdBu_r', aspect='equal', origin='lower')
    axes[0, 1].plot([X.min(), X.max()], [ztab, ztab], 'k--', linewidth=1)
    axes[0, 1].plot(pt.pos[0], pt.pos[2], 'ko', markersize=8)
    axes[0, 1].set_title('|E_z| component')
    axes[0, 1].set_xlabel('x (nm)')
    axes[0, 1].set_ylabel('z (nm)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Field profiles
    # Vertical line through dipole
    dipole_x_idx = np.argmin(np.abs(X[0, :] - pt.pos[0]))
    z_line = Z[:, dipole_x_idx]
    field_z_line = ee[:, dipole_x_idx]
    
    # Horizontal line at dipole height
    dipole_z_idx = np.argmin(np.abs(Z[:, 0] - pt.pos[2]))
    x_line = X[dipole_z_idx, :]
    field_x_line = ee[dipole_z_idx, :]
    
    axes[1, 0].semilogy(z_line, field_z_line, 'b-', linewidth=2)
    axes[1, 0].axvline(x=ztab, color='k', linestyle='--', alpha=0.5, label='Substrate')
    axes[1, 0].axvline(x=pt.pos[2], color='m', linestyle=':', alpha=0.7, label='Dipole')
    axes[1, 0].set_xlabel('z (nm)')
    axes[1, 0].set_ylabel('|E|')
    axes[1, 0].set_title('Field profile along z')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Radiation pattern (polar plot)
    ax_polar = plt.subplot(224, projection='polar')
    ax_polar.plot(theta.flatten(), s / np.max(s), 'r-', linewidth=2)
    ax_polar.set_title('Radiation pattern (quasistatic)')
    ax_polar.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()


def compare_substrate_effects(X, Z, ee, ztab):
    """Compare field enhancement above and below substrate"""
    
    print("\n=== Substrate Effects in Quasistatic Regime ===")
    
    # Divide regions
    above_mask = Z > ztab + 1  # 1 nm above substrate
    below_mask = Z < ztab - 1  # 1 nm below substrate
    interface_mask = np.abs(Z - ztab) < 0.5  # Near interface
    
    regions = [
        ("Above substrate", above_mask),
        ("Below substrate", below_mask), 
        ("Near interface", interface_mask)
    ]
    
    for region_name, mask in regions:
        if np.any(mask):
            region_field = ee[mask]
            print(f"{region_name}:")
            print(f"  Mean field: {np.mean(region_field):.2f}")
            print(f"  Max field: {np.max(region_field):.2f}")
            print(f"  Field range: {np.min(region_field):.2f} - {np.max(region_field):.2f}")
    
    # Enhancement due to substrate
    if np.any(above_mask) and np.any(below_mask):
        enhancement_ratio = np.mean(ee[above_mask]) / np.mean(ee[below_mask])
        print(f"\nSubstrate enhancement factor: {enhancement_ratio:.2f}")
        
        if enhancement_ratio > 1:
            print("Fields are enhanced above the substrate")
        else:
            print("Fields are stronger below the substrate")
    
    print(f"\nQuasistatic approximation characteristics:")
    print(f"- No retardation effects")
    print(f"- Substrate acts as image source")
    print(f"- Local field enhancement dominates")
    print(f"- Radiation pattern includes substrate reflections")


if __name__ == '__main__':
    e_total, ee, s, theta, X, Z, pt = main()
    
    print("\n=== Simulation Summary ===")
    print(f"System: 4 nm Au nanosphere 1 nm above glass substrate")
    print(f"Dipole: z-oriented at ({pt.pos[0]}, {pt.pos[1]}, {pt.pos[2]}) nm")
    print(f"Wavelength: 550 nm")
    print(f"Method: Quasistatic approximation with substrate")
    print(f"Field grid: {X.shape[0]} × {X.shape[1]} points")
    print(f"Substrate interface at z = 0 nm")
    
    # Detailed analysis
    plot_detailed_quasistatic_analysis(X, Z, e_total, ee, theta, s, pt, 0)
    compare_substrate_effects(X, Z, ee, 0)