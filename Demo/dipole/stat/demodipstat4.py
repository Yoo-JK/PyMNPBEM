"""
DEMODIPSTAT4 - Photonic LDOS for a silver nanodisk.

For a silver nanodisk we compute the photonic LDOS for different
positions and energies within the quasistatic approximation. This
program shows how to produce a nanodisk with a refined discretization
for LDOS simulations.

Runtime: ~37 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import polygon, tripolygon, comparticle
from pymnpbem.mesh2d import edgeprofile
from pymnpbem.misc import compoint, units
from pymnpbem.bem import dipole, bemsolver


def refinement_function(pos, d=None):
    """Refinement function for mesh"""
    return 0.5 + np.abs(pos[:, 1]) ** 2 + 5 * (pos[:, 0] < 0)


def main():
    # Initialization
    # Options for BEM simulation (quasistatic)
    op = bemoptions(sim='stat', interp='curv')
    
    # Table of dielectric functions
    epstab = [epsconst(1), epstable('silver.dat')]
    
    # Polygon for disk
    poly = polygon(25, size=[30, 30])
    
    # Edge profile for disk
    edge = edgeprofile(5, 11)
    
    # Extrude polygon to nanoparticle with refinement
    p, poly = tripolygon(poly, edge, refun=refinement_function)
    
    # Set up COMPARTICLE object
    p = comparticle(epstab, [p], [2, 1], 1, op)
    
    # Dipole oscillator
    # Transition energies (eV)
    ene = np.linspace(1.5, 4, 60)
    
    # Transform to wavelengths
    unit_converter = units()
    enei = unit_converter.eV2nm / ene
    
    # Positions
    x = np.linspace(0, 30, 70).reshape(-1, 1)
    
    # Compoint
    positions = np.column_stack([x.flatten(), np.zeros_like(x.flatten()), 
                                np.full_like(x.flatten(), 3.5)])
    pt = compoint(p, positions)
    
    # Dipole excitation
    dip = dipole(pt, op)
    
    # Initialize total and radiative scattering rate
    tot = np.zeros((len(enei), len(x), 3))
    rad = np.zeros((len(enei), len(x), 3))
    
    # BEM simulation
    # Set up BEM solver with eigenmode expansion to speed up simulation
    bem = bemsolver(p, op, nev=50)
    
    print("Running quasistatic BEM simulation for LDOS mapping...")
    
    # Loop over wavelengths
    for ien in tqdm(range(len(enei)), desc="BEM solver", ncols=80):
        # Surface charge
        sig = bem.solve(dip(p, enei[ien]))
        
        # Total and radiative decay rate
        tot[ien, :, :], rad[ien, :, :] = dip.decayrate(sig)
    
    print("LDOS simulation completed!")
    
    # Final plot
    # Density plot of total scattering rate (LDOS)
    ldos_total = np.sum(tot, axis=2)  # Sum over polarizations
    
    plt.figure(figsize=(12, 8))
    
    # Create the density plot
    im = plt.imshow(np.log10(ldos_total).T, extent=[ene[0], ene[-1], x[0, 0], x[-1, 0]],
                   aspect='auto', origin='lower', cmap='viridis')
    
    # Plot disk edge
    plt.plot(ene, np.full_like(ene, 15), 'w--', linewidth=2, label='Disk edge')
    
    plt.colorbar(im, label='log₁₀(LDOS)')
    plt.xlabel('Energy (eV)')
    plt.ylabel('x (nm)')
    plt.title('LDOS for silver nanodisk')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    analyze_ldos_distribution(ene, x, tot, rad)
    
    return tot, rad, ene, enei, x


def analyze_ldos_distribution(ene, x, tot, rad):
    """Analyze LDOS distribution characteristics"""
    
    print("\n=== LDOS Distribution Analysis ===")
    
    # Sum over polarizations for total LDOS
    ldos_total = np.sum(tot, axis=2)
    
    # Find global maximum
    max_idx = np.unravel_index(np.argmax(ldos_total), ldos_total.shape)
    max_energy = ene[max_idx[0]]
    max_position = x[max_idx[1], 0]
    max_ldos = ldos_total[max_idx]
    
    print(f"Global LDOS maximum:")
    print(f"  Energy: {max_energy:.2f} eV")
    print(f"  Position: x = {max_position:.1f} nm")
    print(f"  LDOS value: {max_ldos:.2f}")
    
    # Analysis at disk center and edge
    center_idx = np.argmin(np.abs(x[:, 0] - 0))
    edge_idx = np.argmin(np.abs(x[:, 0] - 15))
    
    print(f"\nLDOS at disk center (x = 0 nm):")
    center_ldos = ldos_total[:, center_idx]
    center_max_idx = np.argmax(center_ldos)
    print(f"  Maximum: {center_ldos[center_max_idx]:.2f} at {ene[center_max_idx]:.2f} eV")
    
    print(f"LDOS at disk edge (x = 15 nm):")
    edge_ldos = ldos_total[:, edge_idx]
    edge_max_idx = np.argmax(edge_ldos)
    print(f"  Maximum: {edge_ldos[edge_max_idx]:.2f} at {ene[edge_max_idx]:.2f} eV")
    
    # Find resonance energies (peaks in LDOS)
    print(f"\nResonance analysis:")
    
    # Average LDOS across all positions
    avg_ldos = np.mean(ldos_total, axis=1)
    
    # Find peaks
    peaks = []
    for i in range(2, len(ene)-2):
        if (avg_ldos[i] > avg_ldos[i-1] and avg_ldos[i] > avg_ldos[i+1] and
            avg_ldos[i] > avg_ldos[i-2] and avg_ldos[i] > avg_ldos[i+2]):
            if avg_ldos[i] > 2.0:  # Only significant peaks
                peaks.append((ene[i], avg_ldos[i]))
    
    if peaks:
        for j, (peak_energy, peak_ldos) in enumerate(peaks):
            print(f"  Peak {j+1}: {peak_energy:.2f} eV, LDOS = {peak_ldos:.2f}")
    else:
        max_avg_idx = np.argmax(avg_ldos)
        print(f"  Broadband maximum: {ene[max_avg_idx]:.2f} eV, LDOS = {avg_ldos[max_avg_idx]:.2f}")


def plot_detailed_ldos_analysis(ene, x, tot, rad):
    """Detailed LDOS analysis plots"""
    
    ldos_total = np.sum(tot, axis=2)
    ldos_rad = np.sum(rad, axis=2)
    ldos_nonrad = ldos_total - ldos_rad
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Total LDOS
    im1 = axes[0, 0].imshow(np.log10(ldos_total).T, extent=[ene[0], ene[-1], x[0, 0], x[-1, 0]],
                           aspect='auto', origin='lower', cmap='viridis')
    axes[0, 0].plot(ene, np.full_like(ene, 15), 'w--', linewidth=1)
    axes[0, 0].set_xlabel('Energy (eV)')
    axes[0, 0].set_ylabel('x (nm)')
    axes[0, 0].set_title('Total LDOS (log scale)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Radiative LDOS
    im2 = axes[0, 1].imshow(np.log10(ldos_rad + 1e-10).T, extent=[ene[0], ene[-1], x[0, 0], x[-1, 0]],
                           aspect='auto', origin='lower', cmap='plasma')
    axes[0, 1].plot(ene, np.full_like(ene, 15), 'w--', linewidth=1)
    axes[0, 1].set_xlabel('Energy (eV)')
    axes[0, 1].set_ylabel('x (nm)')
    axes[0, 1].set_title('Radiative LDOS (log scale)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Non-radiative LDOS
    im3 = axes[1, 0].imshow(np.log10(ldos_nonrad + 1e-10).T, extent=[ene[0], ene[-1], x[0, 0], x[-1, 0]],
                           aspect='auto', origin='lower', cmap='inferno')
    axes[1, 0].plot(ene, np.full_like(ene, 15), 'w--', linewidth=1)
    axes[1, 0].set_xlabel('Energy (eV)')
    axes[1, 0].set_ylabel('x (nm)')
    axes[1, 0].set_title('Non-radiative LDOS (log scale)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # LDOS profiles at specific positions
    center_idx = np.argmin(np.abs(x[:, 0] - 0))
    edge_idx = np.argmin(np.abs(x[:, 0] - 15))
    outside_idx = np.argmin(np.abs(x[:, 0] - 25))
    
    axes[1, 1].plot(ene, ldos_total[:, center_idx], 'b-', label='Center (x=0)', linewidth=2)
    axes[1, 1].plot(ene, ldos_total[:, edge_idx], 'r-', label='Edge (x=15)', linewidth=2)
    axes[1, 1].plot(ene, ldos_total[:, outside_idx], 'g-', label='Outside (x=25)', linewidth=2)
    axes[1, 1].set_xlabel('Energy (eV)')
    axes[1, 1].set_ylabel('LDOS')
    axes[1, 1].set_title('LDOS profiles at different positions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()


def plot_polarization_analysis(ene, x, tot):
    """Analyze polarization-resolved LDOS"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    polarizations = ['x', 'y', 'z']
    cmaps = ['Reds', 'Greens', 'Blues']
    
    for i in range(3):
        im = axes[i].imshow(np.log10(tot[:, :, i] + 1e-10).T, 
                           extent=[ene[0], ene[-1], x[0, 0], x[-1, 0]],
                           aspect='auto', origin='lower', cmap=cmaps[i])
        axes[i].plot(ene, np.full_like(ene, 15), 'w--', linewidth=1)
        axes[i].set_xlabel('Energy (eV)')
        axes[i].set_ylabel('x (nm)')
        axes[i].set_title(f'{polarizations[i]}-polarized LDOS')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    tot, rad, ene, enei, x = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Silver nanodisk: 30 nm diameter, 5 nm height")
    print(f"Energy range: {ene[0]:.1f} - {ene[-1]:.1f} eV")
    print(f"Wavelength range: {enei[-1]:.0f} - {enei[0]:.0f} nm")
    print(f"Position range: x = 0 - 30 nm, z = 3.5 nm")
    print(f"Grid: {len(ene)} energies × {len(x)} positions")
    print(f"Method: Quasistatic BEM with eigenmode expansion")
    
    # Show detailed analysis
    plot_detailed_ldos_analysis(ene, x, tot, rad)
    plot_polarization_analysis(ene, x, tot)