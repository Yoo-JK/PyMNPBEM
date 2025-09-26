"""
DEMODIPRET9 - Electric field for dipole between sphere and layer.

For a metallic nanosphere with a diameter of 4 nm located 1 nm above a
substrate and a dipole located between sphere and layer, this program
computes the radiation pattern and the total electric field using the
full Maxwell equations.

Runtime: ~50 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import trispheresegment, comparticle, shift, flip
from pymnpbem.misc import compoint, layerstructure, tabspace, compgreentablayer, meshfield, spectrum, farfield
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
    
    # Options for BEM simulation
    op = bemoptions(sim='ret', interp='curv', layer=layer)
    
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
    
    print("Setting up field calculation points...")
    
    # Dipole position
    pt1 = compoint(p, [1, 0, 0.5], op)  # Between sphere and substrate
    
    # Mesh for calculation of electric field
    x_range = np.linspace(-8, 8, 81)
    z_range = np.linspace(-8, 8, 81)
    X, Z = np.meshgrid(x_range, z_range)
    
    # Make compoint object for field calculation
    # Important: COMPOINT receives OP structure for layer grouping
    field_positions = np.column_stack([X.flatten(), np.zeros(X.size), Z.flatten()])
    pt2 = compoint(p, field_positions, op)
    
    # Wavelength corresponding to transition dipole energy
    enei = 550
    
    print("Computing Green's function table (this may take a while)...")
    
    # Tabulated Green functions
    # For retarded simulation, set up table for reflected Green function
    # This part is computationally expensive
    
    # Automatic grid for tabulation (small NZ for speed)
    tab = tabspace(layer, [p, pt1], pt2, nz=5)
    
    # Green function table
    greentab = compgreentablayer(layer, tab)
    
    # Precompute Green function table
    greentab = greentab.set(enei, op, waitbar=False)
    
    # Add Green table to options
    op.greentab = greentab
    
    print("Green's function table completed!")
    
    # BEM simulation
    print("Running BEM simulation...")
    
    # Dipole excitation (z-oriented)
    dip = dipole(pt1, np.array([[0, 0, 1]]), op)
    
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
    # Object for electric field calculation
    # MINDIST controls minimal distance to particle boundary
    emesh = meshfield(p, X, 0, Z, op, mindist=0.15, nmax=3000)
    
    # Induced and incoming electric field
    e_induced = emesh(sig)
    e_incoming = emesh(dip.field(emesh.pt, enei))
    e_total = e_induced + e_incoming
    
    # Norm of electric field
    ee = np.sqrt(np.sum(np.abs(e_total)**2, axis=2))
    
    print("Creating visualizations...")
    
    # Plot results
    plot_field_and_pattern(X, Z, ee, theta, s, pt1, enei, ztab)
    plot_detailed_analysis(X, Z, e_total, theta, s, pt1, ztab)
    
    return e_total, ee, s, theta, X, Z, pt1


def plot_field_and_pattern(X, Z, ee, theta, s, pt1, wavelength, ztab):
    """Plot electric field map with radiation pattern and substrate interface"""
    
    plt.figure(figsize=(12, 10))
    
    # Plot electric field (logarithmic scale)
    im = plt.imshow(np.log10(ee), extent=[X.min(), X.max(), Z.min(), Z.max()],
                   cmap='hot', aspect='equal', origin='lower')
    
    # Plot substrate interface
    plt.plot([X.min(), X.max()], [ztab, ztab], 'w--', linewidth=2, 
            label='Substrate interface')
    
    # Dipole position
    plt.plot(pt1.pos[0], pt1.pos[2], 'mo', markersize=12, 
            markerfacecolor='m', markeredgecolor='white', linewidth=2, 
            label='Dipole')
    
    # Cartesian coordinates of Poynting vector (radiation pattern)
    sx = 8 * s / np.max(s) * np.cos(theta.flatten())
    sy = 8 * s / np.max(s) * np.sin(theta.flatten())
    
    # Overlay radiation pattern
    plt.plot(sx, sy, 'w-', linewidth=2, label='Radiation pattern')
    
    # Color scale
    plt.clim([-2, 2])
    plt.colorbar(im, label='log₁₀(|E|)')
    
    plt.xlabel('x (nm)')
    plt.ylabel('z (nm)')
    plt.title(f'Electric field (log scale) and radiation pattern (λ = {wavelength} nm)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_detailed_analysis(X, Z, e_total, theta, s, pt1, ztab):
    """Detailed analysis with field components and substrate effects"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Field components above and below substrate
    above_mask = Z > ztab
    below_mask = Z <= ztab
    
    # Total field magnitude
    ee_total = np.sqrt(np.sum(np.abs(e_total)**2, axis=2))
    
    # Above substrate
    ee_above = np.copy(ee_total)
    ee_above[below_mask] = np.nan
    im1 = axes[0, 0].imshow(np.log10(ee_above), extent=[X.min(), X.max(), Z.min(), Z.max()],
                           cmap='hot', aspect='equal', origin='lower')
    axes[0, 0].plot([X.min(), X.max()], [ztab, ztab], 'w--', linewidth=1)
    axes[0, 0].plot(pt1.pos[0], pt1.pos[2], 'mo', markersize=8)
    axes[0, 0].set_title('Electric field above substrate')
    axes[0, 0].set_xlabel('x (nm)')
    axes[0, 0].set_ylabel('z (nm)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Below substrate
    ee_below = np.copy(ee_total)
    ee_below[above_mask] = np.nan
    im2 = axes[0, 1].imshow(np.log10(ee_below), extent=[X.min(), X.max(), Z.min(), Z.max()],
                           cmap='hot', aspect='equal', origin='lower')
    axes[0, 1].plot([X.min(), X.max()], [ztab, ztab], 'w--', linewidth=1)
    axes[0, 1].set_title('Electric field below substrate')
    axes[0, 1].set_xlabel('x (nm)')
    axes[0, 1].set_ylabel('z (nm)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # z-component of electric field (shows substrate reflection effects)
    ez = np.abs(e_total[:, :, 2])
    im3 = axes[1, 0].imshow(np.log10(ez + 1e-10), extent=[X.min(), X.max(), Z.min(), Z.max()],
                           cmap='RdBu_r', aspect='equal', origin='lower')
    axes[1, 0].plot([X.min(), X.max()], [ztab, ztab], 'k--', linewidth=1)
    axes[1, 0].plot(pt1.pos[0], pt1.pos[2], 'ko', markersize=8)
    axes[1, 0].set_title('|E_z| component (log scale)')
    axes[1, 0].set_xlabel('x (nm)')
    axes[1, 0].set_ylabel('z (nm)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Radiation pattern (polar plot)
    ax_polar = plt.subplot(224, projection='polar')
    ax_polar.plot(theta.flatten(), s / np.max(s), 'r-', linewidth=2)
    ax_polar.set_title('Normalized radiation pattern')
    ax_polar.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()


def analyze_substrate_field_effects(X, Z, ee, ztab, pt1):
    """Analyze substrate effects on field distribution"""
    
    print("\n=== Substrate Field Effects Analysis ===")
    
    # Separate regions
    above_mask = Z > ztab
    below_mask = Z <= ztab
    
    # Field statistics above substrate
    ee_above = ee[above_mask]
    max_above = np.max(ee_above)
    mean_above = np.mean(ee_above)
    
    # Field statistics below substrate
    ee_below = ee[below_mask]
    max_below = np.max(ee_below)
    mean_below = np.mean(ee_below)
    
    print(f"Above substrate (air):")
    print(f"  Maximum field: {max_above:.2f}")
    print(f"  Mean field: {mean_above:.2f}")
    
    print(f"Below substrate (glass):")
    print(f"  Maximum field: {max_below:.2f}")
    print(f"  Mean field: {mean_below:.2f}")
    
    print(f"Field ratio (above/below): {max_above/max_below:.2f}")
    
    # Field enhancement at dipole region
    dipole_x_idx = np.argmin(np.abs(X[0, :] - pt1.pos[0]))
    dipole_z_idx = np.argmin(np.abs(Z[:, 0] - pt1.pos[2]))
    
    local_field = ee[max(0, dipole_z_idx-2):dipole_z_idx+3, 
                     max(0, dipole_x_idx-2):dipole_x_idx+3]
    local_enhancement = np.mean(local_field)
    
    print(f"Local field enhancement around dipole: {local_enhancement:.2f}")


def analyze_radiation_asymmetry(theta, s):
    """Analyze radiation pattern asymmetry due to substrate"""
    
    print("\n=== Radiation Pattern Analysis ===")
    
    # Normalize
    s_norm = s / np.max(s)
    
    # Split into upper and lower hemisphere
    theta_deg = np.degrees(theta.flatten())
    upper_mask = (theta_deg >= 0) & (theta_deg <= 180)
    lower_mask = (theta_deg > 180) & (theta_deg <= 360)
    
    # Power radiated in each hemisphere
    upper_power = np.trapz(s_norm[upper_mask], theta_deg[upper_mask])
    lower_power = np.trapz(s_norm[lower_mask], theta_deg[lower_mask])
    
    total_power = upper_power + lower_power
    
    print(f"Power radiated into upper hemisphere: {upper_power/total_power*100:.1f}%")
    print(f"Power radiated into lower hemisphere: {lower_power/total_power*100:.1f}%")
    print(f"Asymmetry ratio (upper/lower): {upper_power/lower_power:.2f}")
    
    # Find main radiation directions
    max_idx = np.argmax(s_norm)
    main_direction = theta_deg[max_idx]
    
    print(f"Main radiation direction: {main_direction:.1f}°")


if __name__ == '__main__':
    e_total, ee, s, theta, X, Z, pt1 = main()
    
    print("\n=== Simulation Summary ===")
    print(f"System: 4 nm Au nanosphere 1 nm above glass substrate")
    print(f"Dipole: z-oriented at ({pt1.pos[0]}, {pt1.pos[1]}, {pt1.pos[2]}) nm")
    print(f"Wavelength: 550 nm")
    print(f"Field grid: {X.shape[0]} × {X.shape[1]} points")
    print(f"Radiation pattern: {len(theta)} angular points")
    
    # Analyze results
    analyze_substrate_field_effects(X, Z, ee, 0, pt1)
    analyze_radiation_asymmetry(theta, s)