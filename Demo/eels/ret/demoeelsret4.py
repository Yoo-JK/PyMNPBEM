"""
DEMOEELSRET4 - EELS of nanotriangle for different impact parameters.

For a silver nanotriangle with 80 nm base length and 10 nm height,
this program computes the energy loss probability for selected impact
parameters and loss energies using the full Maxwell equations.

Runtime: ~66 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import polygon, tripolygon, comparticle
from pymnpbem.mesh2d import edgeprofile
from pymnpbem.misc import units, eelsbase
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
    hdata = {'hmax': 8}
    
    # Extrude polygon
    p = tripolygon(poly, edge, hdata=hdata)
    
    # Make particle
    p = comparticle(epstab, [p], [2, 1], 1, op)
    
    # Width of electron beam and electron velocity
    width = 0.2
    vel = eelsbase.ene2vel(200e3)  # 200 keV electron energy
    
    # Impact parameters (triangle corner, middle, and midpoint of edge)
    imp = np.array([[-45, 0], [0, 0], [25, 0]])
    
    # Loss energies in eV
    ene = np.linspace(1.5, 4.5, 40)
    
    # Convert energies to nm
    unit_converter = units()
    enei = unit_converter.eV2nm / ene
    
    # BEM simulation
    # BEM solver
    bem = bemsolver(p, op)
    
    # Electron beam excitation
    exc = electronbeam(p, imp, width, vel, op)
    
    # Surface and bulk loss
    psurf = np.zeros((imp.shape[0], len(enei)))
    pbulk = np.zeros((imp.shape[0], len(enei)))
    
    print("Running BEM simulation for nanotriangle EELS...")
    
    # Loop over energies
    for ien in tqdm(range(len(ene)), desc="BEM solver", ncols=80):
        # Surface charges
        sig = bem.solve(exc(enei[ien]))
        
        # EELS losses
        psurf[:, ien], pbulk[:, ien] = exc.loss(sig)
    
    print("Nanotriangle EELS simulation completed!")
    
    # Total losses
    ptotal = psurf + pbulk
    
    # Final plot
    plt.figure(figsize=(12, 8))
    
    # Plot total losses for each impact parameter
    impact_labels = ['Corner', 'Middle', 'Edge']
    colors = ['red', 'blue', 'green']
    
    for i in range(imp.shape[0]):
        plt.plot(ene, ptotal[i, :], color=colors[i], 
                label=impact_labels[i], linewidth=2, marker='o', markersize=4)
    
    plt.legend()
    plt.xlabel('Loss energy (eV)')
    plt.ylabel('Loss probability (eV⁻¹)')
    plt.title('EELS of silver nanotriangle (80 nm base, 10 nm height)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    analyze_triangle_eels(ene, ptotal, psurf, pbulk, imp)
    
    return ptotal, psurf, pbulk, ene, imp


def analyze_triangle_eels(ene, ptotal, psurf, pbulk, imp):
    """Analyze nanotriangle EELS characteristics"""
    
    print("\n=== Nanotriangle EELS Analysis ===")
    
    impact_labels = ['Corner (-45, 0) nm', 'Middle (0, 0) nm', 'Edge (25, 0) nm']
    
    for i in range(len(impact_labels)):
        print(f"\n{impact_labels[i]}:")
        
        # Find peak energy and intensity
        peak_idx = np.argmax(ptotal[i, :])
        peak_energy = ene[peak_idx]
        peak_intensity = ptotal[i, peak_idx]
        
        print(f"  Peak energy: {peak_energy:.2f} eV")
        print(f"  Peak intensity: {peak_intensity:.3e} eV⁻¹")
        
        # Surface vs bulk contribution analysis
        avg_surf_contribution = np.mean(psurf[i, :] / (ptotal[i, :] + 1e-15))
        avg_bulk_contribution = np.mean(pbulk[i, :] / (ptotal[i, :] + 1e-15))
        
        print(f"  Average surface contribution: {avg_surf_contribution*100:.1f}%")
        print(f"  Average bulk contribution: {avg_bulk_contribution*100:.1f}%")
        
        # Energy range analysis
        significant_range = ene[ptotal[i, :] > 0.1 * peak_intensity]
        if len(significant_range) > 0:
            print(f"  Active energy range: {significant_range[0]:.1f} - {significant_range[-1]:.1f} eV")
    
    # Compare impact parameters
    print(f"\nImpact parameter comparison:")
    max_intensities = [np.max(ptotal[i, :]) for i in range(3)]
    peak_energies = [ene[np.argmax(ptotal[i, :])] for i in range(3)]
    
    max_position = np.argmax(max_intensities)
    print(f"Strongest response: {impact_labels[max_position]}")
    print(f"Peak energies: Corner {peak_energies[0]:.2f} eV, Middle {peak_energies[1]:.2f} eV, Edge {peak_energies[2]:.2f} eV")


def plot_detailed_triangle_analysis(ene, ptotal, psurf, pbulk, imp):
    """Detailed analysis plots for nanotriangle EELS"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    impact_labels = ['Corner', 'Middle', 'Edge']
    colors = ['red', 'blue', 'green']
    
    # Total loss probability
    for i in range(3):
        axes[0, 0].plot(ene, ptotal[i, :], color=colors[i], 
                       label=impact_labels[i], linewidth=2)
    axes[0, 0].set_xlabel('Loss energy (eV)')
    axes[0, 0].set_ylabel('Total loss probability (eV⁻¹)')
    axes[0, 0].set_title('Total EELS Response')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Surface contributions
    for i in range(3):
        axes[0, 1].plot(ene, psurf[i, :], color=colors[i], 
                       label=f'{impact_labels[i]} (surf)', linewidth=2)
    axes[0, 1].set_xlabel('Loss energy (eV)')
    axes[0, 1].set_ylabel('Surface loss probability (eV⁻¹)')
    axes[0, 1].set_title('Surface Contributions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bulk contributions
    for i in range(3):
        axes[1, 0].plot(ene, pbulk[i, :], color=colors[i], 
                       label=f'{impact_labels[i]} (bulk)', linewidth=2)
    axes[1, 0].set_xlabel('Loss energy (eV)')
    axes[1, 0].set_ylabel('Bulk loss probability (eV⁻¹)')
    axes[1, 0].set_title('Bulk Contributions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Surface/bulk ratio
    for i in range(3):
        ratio = psurf[i, :] / (pbulk[i, :] + 1e-15)
        axes[1, 1].plot(ene, ratio, color=colors[i], 
                       label=impact_labels[i], linewidth=2)
    axes[1, 1].axhline(y=1, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Loss energy (eV)')
    axes[1, 1].set_ylabel('Surface/Bulk ratio')
    axes[1, 1].set_title('Surface vs Bulk Dominance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()


def plot_triangle_geometry_effects(ene, ptotal, imp):
    """Plot showing geometry-dependent effects"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    impact_labels = ['Corner', 'Middle', 'Edge']
    colors = ['red', 'blue', 'green']
    
    # Normalized comparison
    axes[0].set_title('Normalized EELS Response')
    for i in range(3):
        ptotal_norm = ptotal[i, :] / np.max(ptotal[i, :])
        axes[0].plot(ene, ptotal_norm, color=colors[i], 
                    label=impact_labels[i], linewidth=2)
    
    axes[0].set_xlabel('Loss energy (eV)')
    axes[0].set_ylabel('Normalized loss probability')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Peak shift analysis
    peak_energies = []
    peak_intensities = []
    
    for i in range(3):
        peak_idx = np.argmax(ptotal[i, :])
        peak_energies.append(ene[peak_idx])
        peak_intensities.append(ptotal[i, peak_idx])
    
    impact_positions = ['Corner\n(-45,0)', 'Middle\n(0,0)', 'Edge\n(25,0)']
    
    axes[1].bar(impact_positions, peak_intensities, color=colors, alpha=0.7)
    axes[1].set_ylabel('Peak intensity (eV⁻¹)')
    axes[1].set_title('Peak Intensity vs Impact Position')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add peak energy labels on bars
    for i, (pos, energy) in enumerate(zip(impact_positions, peak_energies)):
        axes[1].text(i, peak_intensities[i] * 1.05, f'{energy:.1f} eV', 
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def explain_triangle_eels_physics():
    """Explain triangle EELS physics"""
    
    print("\n=== Triangle EELS Physics ===")
    print("Impact parameter effects in triangular geometry:")
    print("- Corner: High field concentration, strong multipolar excitation")
    print("- Middle: Symmetric excitation of fundamental modes")
    print("- Edge: Dipolar surface plasmon dominance")
    print("- Sharp corners enhance local fields")
    print("- Different modes have different spatial distributions")
    print("- Triangle symmetry breaks degeneracies")


if __name__ == '__main__':
    ptotal, psurf, pbulk, ene, imp = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Silver nanotriangle: 80 nm base length, 10 nm height")
    print(f"Electron beam: 200 keV energy, 0.2 nm width")
    print(f"Impact parameters: Corner (-45,0), Middle (0,0), Edge (25,0) nm")
    print(f"Loss energy range: {ene[0]:.1f} - {ene[-1]:.1f} eV")
    print(f"Calculation: Full retarded BEM")
    
    # Detailed analysis
    plot_detailed_triangle_analysis(ene, ptotal, psurf, pbulk, imp)
    plot_triangle_geometry_effects(ene, ptotal, imp)
    explain_triangle_eels_physics()