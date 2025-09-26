"""
DEMODIPRET11 - Electric field for dipole close to nanodisk and layer.

For a silver nanodisk with a diameter of 30 nm and a height of 5 nm,
and a dipole oscillator with dipole orientation along x and located 5
nm away from the disk and 0.5 nm above the substrate, this program
computes the radiation pattern and the total electric field using the
full Maxwell equations.

Runtime: ~32 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import polygon, tripolygon, comparticle
from pymnpbem.mesh2d import edgeprofile
from pymnpbem.misc import compoint, layerstructure, tabspace, compgreentablayer, meshfield, spectrum, farfield
from pymnpbem.bem import dipole, bemsolver


def refinement_function(pos, d):
    """Refinement function for finer discretization at (R,0,0)"""
    return 0.5 + np.abs(15 - pos[:, 0])


def main():
    # Initialization
    # Table of dielectric functions
    epstab = [epsconst(1), epstable('silver.dat'), epsconst(4)]  # air, silver, substrate
    
    # Location of interface of substrate
    ztab = 0
    
    # Default options for layer structure
    opt = layerstructure.options()
    
    # Set up layer structure
    layer = layerstructure(epstab, [1, 3], ztab, opt)  # air above, substrate below
    
    # Options for BEM simulation (quasistatic)
    op = bemoptions(sim='stat', interp='curv', layer=layer)
    
    # Polygon for disk
    poly = polygon(25, size=[30, 30])
    
    # Edge profile for nanodisk
    # MODE '01' produces rounded edge on top and sharp edge on bottom
    edge = edgeprofile(5, 11, mode='01', min=1e-3)
    
    # Extrude polygon to nanoparticle
    p = tripolygon(poly, edge, refun=refinement_function)
    
    # Set up COMPARTICLE object
    p = comparticle(epstab, [p], [2, 1], 1, op)  # silver disk in air
    
    print("Setting up field calculation points...")
    
    # Position of dipole
    x_dipole = np.max(p.pos[:, 0]) + 2  # 2 nm from edge
    
    # Compoint for dipole
    pt1 = compoint(p, [x_dipole, 0, 0.5], op)  # 0.5 nm above substrate
    
    # Mesh for calculation of electric field
    x_range = np.linspace(-30, 30, 81)
    z_range = np.linspace(-30, 30, 81)
    X, Z = np.meshgrid(x_range, z_range)
    
    # Make compoint object for field calculation
    # Important: COMPOINT receives OP structure for layer grouping
    field_positions = np.column_stack([X.flatten(), np.zeros(X.size), Z.flatten()])
    pt2 = compoint(p, field_positions, op)
    
    # Wavelength corresponding to transition dipole energy
    enei = 620
    
    print("Computing Green's function table...")
    
    # Tabulated Green functions
    # Automatic grid for tabulation (small NZ for speed)
    tab = tabspace(layer, [p, pt1], pt2, nz=5)
    
    # Green function table
    greentab = compgreentablayer(layer, tab)
    
    # Precompute Green function table
    greentab = greentab.set(enei, op, waitbar=False)
    
    # Add Green table to options
    op.greentab = greentab
    
    print("Green's function table completed!")
    
    # Dipole excitation (x-oriented)
    dip = dipole(pt1, np.array([[1, 0, 0]]), op)
    
    print("Running BEM simulation...")
    
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
    # Object for electric field calculation
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
            label='Dipole (x-pol)')
    
    # Cartesian coordinates of Poynting vector (radiation pattern)
    sx = 20 * s / np.max(s) * np.cos(theta.flatten())
    sy = 20 * s / np.max(s) * np.sin(theta.flatten())
    
    # Overlay radiation pattern
    plt.plot(sx, sy, 'w-', linewidth=2, label='Radiation pattern')
    
    # Color scale
    plt.clim([-4, 0])
    plt.colorbar(im, label='log₁₀(|E|)')
    
    plt.xlabel('x (nm)')
    plt.ylabel('z (nm)')
    plt.title(f'Electric field (log scale) and radiation pattern (λ = {wavelength} nm, quasistatic)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_detailed_analysis(X, Z, e_total, theta, s, pt1, ztab):
    """Detailed analysis with field components and substrate effects"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Total field magnitude
    ee_total = np.sqrt(np.sum(np.abs(e_total)**2, axis=2))
    
    # Above and below substrate
    above_mask = Z > ztab
    below_mask = Z <= ztab
    
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
    
    # x-component of electric field (dipole polarization direction)
    ex = np.abs(e_total[:, :, 0])
    im3 = axes[1, 0].imshow(np.log10(ex + 1e-10), extent=[X.min(), X.max(), Z.min(), Z.max()],
                           cmap='RdBu_r', aspect='equal', origin='lower')
    axes[1, 0].plot([X.min(), X.max()], [ztab, ztab], 'k--', linewidth=1)
    axes[1, 0].plot(pt1.pos[0], pt1.pos[2], 'ko', markersize=8)
    axes[1, 0].set_title('|E_x| component (log scale)')
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


def analyze_quasistatic_effects(X, Z, ee, pt1, ztab):
    """Analyze quasistatic approximation effects near nanodisk"""
    
    print("\n=== Quasistatic Field Enhancement Analysis ===")
    
    # Find maximum field enhancement
    max_idx = np.unravel_index(np.argmax(ee), ee.shape)
    max_field = ee[max_idx]
    max_x = X[max_idx]
    max_z = Z[max_idx]
    
    # Distance from dipole to maximum field point
    dist_to_max = np.sqrt((max_x - pt1.pos[0])**2 + (max_z - pt1.pos[2])**2)
    
    print(f"Maximum field enhancement: {max_field:.2f}")
    print(f"Location of maximum: ({max_x:.1f}, {max_z:.1f}) nm")
    print(f"Distance from dipole: {dist_to_max:.1f} nm")
    
    # Field enhancement near nanodisk edge
    # Find points near the disk edge (around x ≈ 15 nm)
    disk_edge_mask = (np.abs(X - 15) < 2) & (np.abs(Z) < 10)
    if np.any(disk_edge_mask):
        edge_field_avg = np.mean(ee[disk_edge_mask])
        print(f"Average field near disk edge: {edge_field_avg:.2f}")
    
    # Field enhancement at substrate interface
    interface_mask = (np.abs(Z - ztab) < 1) & (np.abs(X) < 20)
    if np.any(interface_mask):
        interface_field_avg = np.mean(ee[interface_mask])
        print(f"Average field at substrate interface: {interface_field_avg:.2f}")
    
    # Compare above vs below substrate
    above_mask = Z > ztab + 1  # 1 nm above
    below_mask = Z < ztab - 1  # 1 nm below
    
    if np.any(above_mask) and np.any(below_mask):
        above_avg = np.mean(ee[above_mask])
        below_avg = np.mean(ee[below_mask])
        print(f"Field ratio (above/below substrate): {above_avg/below_avg:.2f}")


def analyze_nanodisk_coupling(X, Z, ee, pt1):
    """Analyze coupling between dipole and nanodisk"""
    
    print("\n=== Nanodisk-Dipole Coupling Analysis ===")
    
    # Field along line between dipole and disk center
    y_center_idx = np.argmin(np.abs(Z[:, 0] - pt1.pos[2]))  # Same height as dipole
    x_line = X[y_center_idx, :]
    field_line = ee[y_center_idx, :]
    
    # Find disk boundaries (approximate)
    disk_left = -15  # nm
    disk_right = 15   # nm
    dipole_x = pt1.pos[0]
    
    # Field at different positions
    dipole_x_idx = np.argmin(np.abs(x_line - dipole_x))
    disk_edge_idx = np.argmin(np.abs(x_line - disk_right))
    disk_center_idx = np.argmin(np.abs(x_line - 0))
    
    print(f"Field at dipole position: {field_line[dipole_x_idx]:.2f}")
    print(f"Field at disk edge: {field_line[disk_edge_idx]:.2f}")
    print(f"Field at disk center: {field_line[disk_center_idx]:.2f}")
    
    # Field decay from dipole
    gap_region = (x_line > disk_right) & (x_line < dipole_x)
    if np.any(gap_region):
        gap_field = field_line[gap_region]
        gap_x = x_line[gap_region]
        
        # Simple exponential fit would require more sophisticated analysis
        print(f"Average field in gap region: {np.mean(gap_field):.2f}")
        print(f"Gap distance: {dipole_x - disk_right:.1f} nm")


if __name__ == '__main__':
    e_total, ee, s, theta, X, Z, pt1 = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Silver nanodisk: 30 nm diameter, 5 nm height on substrate (ε = 4)")
    print(f"Dipole: x-oriented at ({pt1.pos[0]:.1f}, {pt1.pos[1]:.1f}, {pt1.pos[2]:.1f}) nm")
    print(f"Wavelength: 620 nm (quasistatic approximation)")
    print(f"Field grid: {X.shape[0]} × {X.shape[1]} points")
    print(f"Distance from disk edge: {pt1.pos[0] - 15:.1f} nm")
    
    # Analyze results
    analyze_quasistatic_effects(X, Z, ee, pt1, 0)
    analyze_nanodisk_coupling(X, Z, ee, pt1)