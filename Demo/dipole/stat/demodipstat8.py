"""
DEMODIPSTAT8 - Electric field for dipole close to nanodisk and layer.

For a silver nanodisk with a diameter of 30 nm and a height of 5 nm,
and a dipole oscillator with dipole orientation along x and located 5
nm away from the disk and 0.5 nm above the substrate, this program
computes the radiation pattern and the total electric field using the
quasistatic approximation.

Runtime: ~23 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import polygon, tripolygon, comparticle
from pymnpbem.mesh2d import edgeprofile
from pymnpbem.misc import compoint, layerstructure, meshfield, spectrum, farfield
from pymnpbem.bem import dipole, bemsolver


def refinement_function(pos, d=None):
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
    
    # Options for BEM simulation (quasistatic with layer)
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
    
    # Dipole oscillator
    enei = 570
    
    # Position of dipole
    x_dipole = np.max(p.pos[:, 0]) + 2  # 2 nm from edge
    
    # Compoint
    pt = compoint(p, [x_dipole, 0, 0.5], op)  # 0.5 nm above substrate
    
    # Dipole excitation (x-oriented)
    dip = dipole(pt, np.array([[1, 0, 0]]), op)
    
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
    x_range = np.linspace(-30, 30, 81)
    z_range = np.linspace(-30, 30, 81)
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
    sx = 20 * s / np.max(s) * np.cos(theta.flatten())
    sy = 20 * s / np.max(s) * np.sin(theta.flatten())
    
    # Overlay radiation pattern
    plt.plot(sx, sy, 'w-', linewidth=2)
    
    # Color scale
    plt.clim([-3, 1])
    plt.colorbar(im, label='log₁₀(|E|)')
    
    plt.xlabel('x (nm)')
    plt.ylabel('z (nm)')
    plt.title('Electric field (logarithmic), radiation pattern (quasistatic)')
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    analyze_nanodisk_field_coupling(X, Z, e_total, ee, theta, s, pt, ztab)
    
    return e_total, ee, s, theta, X, Z, pt


def analyze_nanodisk_field_coupling(X, Z, e_total, ee, theta, s, pt, ztab):
    """Analyze field coupling between dipole and nanodisk"""
    
    print("\n=== Nanodisk-Dipole Field Coupling Analysis ===")
    
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
    
    # Field enhancement near disk edges
    disk_center = np.array([0, 0])  # x, z coordinates
    disk_radius = 15  # nm
    
    # Points near disk edge
    disk_distance = np.sqrt(X**2 + Z**2)
    near_disk_mask = (disk_distance > disk_radius - 2) & (disk_distance < disk_radius + 2) & (Z > -3) & (Z < 8)
    
    if np.any(near_disk_mask):
        disk_edge_field_avg = np.mean(ee[near_disk_mask])
        disk_edge_field_max = np.max(ee[near_disk_mask])
        print(f"Field near disk edge: avg = {disk_edge_field_avg:.2f}, max = {disk_edge_field_max:.2f}")
    
    # Gap region analysis (between dipole and disk)
    gap_mask = (X > disk_radius) & (X < pt.pos[0]) & (np.abs(Z - pt.pos[2]) < 2)
    if np.any(gap_mask):
        gap_field_avg = np.mean(ee[gap_mask])
        print(f"Average field in gap region: {gap_field_avg:.2f}")
    
    # Substrate enhancement
    above_substrate = Z > ztab + 1
    below_substrate = Z < ztab - 1
    
    if np.any(above_substrate) and np.any(below_substrate):
        above_avg = np.mean(ee[above_substrate])
        below_avg = np.mean(ee[below_substrate])
        print(f"Substrate enhancement ratio: {above_avg/below_avg:.2f}")
    
    # Radiation pattern characteristics
    print(f"\nRadiation pattern analysis:")
    s_norm = s / np.max(s)
    
    # Find radiation maxima
    peaks = []
    for i in range(1, len(s_norm)-1):
        if s_norm[i] > s_norm[i-1] and s_norm[i] > s_norm[i+1] and s_norm[i] > 0.6:
            angle_deg = np.degrees(theta[i, 0])
            peaks.append((angle_deg, s_norm[i]))
    
    if peaks:
        print(f"Radiation peaks:")
        for i, (angle, intensity) in enumerate(peaks[:3]):
            print(f"  Peak {i+1}: θ = {angle:.1f}°, Intensity = {intensity:.3f}")
    
    # Directivity estimate
    total_power = np.trapz(s_norm, theta.flatten())
    max_power = np.max(s_norm)
    directivity = 2 * np.pi * max_power / total_power
    print(f"Approximate directivity: {directivity:.2f}")


def plot_detailed_nanodisk_analysis(X, Z, e_total, ee, theta, s, pt, ztab):
    """Detailed analysis plots for nanodisk system"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Total field with nanodisk outline
    im1 = axes[0, 0].imshow(np.log10(ee), extent=[X.min(), X.max(), Z.min(), Z.max()],
                           cmap='hot', aspect='equal', origin='lower')
    axes[0, 0].plot([X.min(), X.max()], [ztab, ztab], 'w--', linewidth=1)
    axes[0, 0].plot(pt.pos[0], pt.pos[2], 'mo', markersize=8)
    # Add disk outline
    disk_theta = np.linspace(0, 2*np.pi, 100)
    disk_x = 15 * np.cos(disk_theta)  # 15 nm radius
    disk_z_top = 5 * np.ones_like(disk_theta)  # 5 nm height
    disk_z_bottom = np.zeros_like(disk_theta)
    axes[0, 0].plot(disk_x, disk_z_top, 'w-', linewidth=1, alpha=0.7)
    axes[0, 0].plot(disk_x, disk_z_bottom, 'w-', linewidth=1, alpha=0.7)
    axes[0, 0].set_title('Electric field with nanodisk')
    axes[0, 0].set_xlabel('x (nm)')
    axes[0, 0].set_ylabel('z (nm)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # x-component (dipole orientation)
    ex = np.abs(e_total[:, :, 0])
    im2 = axes[0, 1].imshow(np.log10(ex + 1e-10), extent=[X.min(), X.max(), Z.min(), Z.max()],
                           cmap='RdBu_r', aspect='equal', origin='lower')
    axes[0, 1].plot([X.min(), X.max()], [ztab, ztab], 'k--', linewidth=1)
    axes[0, 1].plot(pt.pos[0], pt.pos[2], 'ko', markersize=8)
    axes[0, 1].plot(disk_x, disk_z_top, 'k-', linewidth=1, alpha=0.7)
    axes[0, 1].plot(disk_x, disk_z_bottom, 'k-', linewidth=1, alpha=0.7)
    axes[0, 1].set_title('|E_x| component (dipole direction)')
    axes[0, 1].set_xlabel('x (nm)')
    axes[0, 1].set_ylabel('z (nm)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Field profile along line from dipole to disk center
    # Line from dipole to disk center
    dipole_pos = np.array([pt.pos[0], pt.pos[2]])
    disk_center = np.array([0, 2.5])  # center of disk
    
    # Create line points
    line_points = np.linspace(0, 1, 50)
    line_x = dipole_pos[0] + line_points * (disk_center[0] - dipole_pos[0])
    line_z = dipole_pos[1] + line_points * (disk_center[1] - dipole_pos[1])
    
    # Interpolate field values along line
    from scipy.interpolate import RegularGridInterpolator
    grid_points = (Z[:, 0], X[0, :])
    interpolator = RegularGridInterpolator(grid_points, ee, bounds_error=False, fill_value=0)
    line_field = interpolator(np.column_stack([line_z, line_x]))
    line_distance = np.sqrt((line_x - dipole_pos[0])**2 + (line_z - dipole_pos[1])**2)
    
    axes[1, 0].semilogy(line_distance, line_field, 'b-', linewidth=2)
    axes[1, 0].axvline(x=0, color='m', linestyle=':', alpha=0.7, label='Dipole')
    axes[1, 0].axvline(x=np.max(line_distance), color='silver', linestyle=':', alpha=0.7, label='Disk center')
    axes[1, 0].set_xlabel('Distance from dipole (nm)')
    axes[1, 0].set_ylabel('|E|')
    axes[1, 0].set_title('Field profile: dipole to disk center')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Radiation pattern (polar plot)
    ax_polar = plt.subplot(224, projection='polar')
    ax_polar.plot(theta.flatten(), s / np.max(s), 'r-', linewidth=2)
    ax_polar.set_title('Radiation pattern (quasistatic)')
    ax_polar.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()


def summarize_quasistatic_advantages():
    """Summarize advantages of quasistatic approximation"""
    
    print("\n=== Quasistatic Approximation Summary ===")
    print("Advantages:")
    print("- Fast computation (no retardation integrals)")
    print("- Good accuracy for small particles (size << wavelength)")
    print("- Captures essential near-field physics")
    print("- Substrate effects included through Green's functions")
    print("- Suitable for LDOS and field enhancement calculations")
    
    print("\nLimitations:")
    print("- No propagating wave effects")
    print("- Limited accuracy for large particles")
    print("- Radiation patterns are approximate")
    print("- No frequency dispersion in far-field")
    
    print("\nBest applications:")
    print("- Near-field enhancement studies")
    print("- LDOS calculations for quantum emitters")
    print("- Plasmon resonance analysis")
    print("- Parameter optimization studies")


if __name__ == '__main__':
    e_total, ee, s, theta, X, Z, pt = main()
    
    print("\n=== Final Simulation Summary ===")
    print(f"Silver nanodisk: 30 nm diameter, 5 nm height on substrate")
    print(f"Dipole: x-oriented at ({pt.pos[0]:.1f}, {pt.pos[1]:.1f}, {pt.pos[2]:.1f}) nm")
    print(f"Wavelength: 570 nm")
    print(f"Method: Quasistatic BEM with substrate")
    print(f"Field grid: {X.shape[0]} × {X.shape[1]} points")
    print(f"Gap distance: ~2 nm from disk edge")
    
    # Detailed analysis
    plot_detailed_nanodisk_analysis(X, Z, e_total, ee, theta, s, pt, 0)
    summarize_quasistatic_advantages()
    
    print(f"\nDemo/dipole/stat 폴더의 모든 예제 변환이 완료되었습니다!")
    print(f"총 8개의 준정적 계산 예제를 Python으로 성공적으로 변환했습니다.")