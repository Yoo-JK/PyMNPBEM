"""
DEMOEELSRET2 - EELS of nanodisk for different impact parameters.

For a silver nanodisk with 30 nm diameter and 5 nm height, this
program computes the energy loss probability for selected impact
parameters and loss energies using the full Maxwell equations, and
compares the results with those obtained within the quasistatic
approximation.

Runtime: ~96 sec.
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
    op1 = bemoptions(sim='ret', interp='curv')    # Retarded
    op2 = bemoptions(sim='stat', interp='curv')   # Quasistatic
    
    # Table of dielectric function
    epstab = [epsconst(1), epstable('silver.dat')]
    
    # Diameter of disk
    diameter = 30
    
    # Polygon for disk
    poly = polygon(25, size=[diameter, diameter])
    
    # Edge profile for disk
    edge = edgeprofile(5, 11)
    
    # Extrude polygon to nanoparticle
    p = comparticle(epstab, [tripolygon(poly, edge)], [2, 1], 1, op1)
    
    # Width of electron beam and electron velocity
    width = 0.5
    vel = eelsbase.ene2vel(200e3)  # 200 keV electron energy
    
    # Impact parameters (0, R/2, R)
    radius = diameter / 2
    imp = np.array([[0, 0], [0.48 * radius, 0], [0.96 * radius, 0]])
    
    # Loss energies in eV
    ene = np.linspace(2.5, 4, 40)
    
    # Convert energies to nm
    unit_converter = units()
    enei = unit_converter.eV2nm / ene
    
    # BEM solution
    # BEM solvers
    bem1 = bemsolver(p, op1)  # Retarded
    bem2 = bemsolver(p, op2)  # Quasistatic
    
    # Electron beam excitation
    exc1 = electronbeam(p, imp, width, vel, op1)
    exc2 = electronbeam(p, imp, width, vel, op2)
    
    # Surface and bulk losses
    psurf1 = np.zeros((imp.shape[0], len(enei)))
    pbulk1 = np.zeros((imp.shape[0], len(enei)))
    psurf2 = np.zeros((imp.shape[0], len(enei)))
    pbulk2 = np.zeros((imp.shape[0], len(enei)))
    
    print("Running BEM simulation for EELS with different impact parameters...")
    
    # Loop over wavelengths
    for ien in tqdm(range(len(enei)), desc="BEM solver", ncols=80):
        # Surface charges
        sig1 = bem1.solve(exc1(enei[ien]))  # Retarded
        sig2 = bem2.solve(exc2(enei[ien]))  # Quasistatic
        
        # EELS losses
        psurf1[:, ien], pbulk1[:, ien] = exc1.loss(sig1)
        psurf2[:, ien], pbulk2[:, ien] = exc2.loss(sig2)
    
    print("EELS simulation completed!")
    
    # Total losses
    ptotal1 = psurf1 + pbulk1  # Retarded
    ptotal2 = psurf2 + pbulk2  # Quasistatic
    
    # Final plot
    plt.figure(figsize=(12, 8))
    
    # Plot retarded results
    impact_labels = ['0', 'R/2', 'R']
    colors = ['red', 'blue', 'green']
    
    for i in range(imp.shape[0]):
        plt.plot(ene, ptotal1[i, :], '-', color=colors[i], 
                label=f'{impact_labels[i]}, ret', linewidth=2)
    
    # Plot quasistatic results
    for i in range(imp.shape[0]):
        plt.plot(ene, ptotal2[i, :], '--', color=colors[i], 
                label=f'{impact_labels[i]}, qs', linewidth=2)
    
    plt.xlabel('Loss energy (eV)')
    plt.ylabel('Loss probability (eV⁻¹)')
    plt.legend()
    plt.title('EELS of silver nanodisk: retarded vs quasistatic')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    analyze_impact_parameter_effects(ene, ptotal1, ptotal2, psurf1, psurf2, 
                                   pbulk1, pbulk2, imp, diameter)
    
    return ptotal1, ptotal2, psurf1, psurf2, pbulk1, pbulk2, ene, imp


def analyze_impact_parameter_effects(ene, ptotal1, ptotal2, psurf1, psurf2, 
                                   pbulk1, pbulk2, imp, diameter):
    """Analyze impact parameter effects on EELS"""
    
    print("\n=== Impact Parameter Effects Analysis ===")
    
    impact_labels = ['Center (b=0)', 'Mid-radius (b=R/2)', 'Edge (b=R)']
    impact_distances = [0, 0.48 * diameter/2, 0.96 * diameter/2]
    
    for i in range(len(impact_labels)):
        print(f"\n{impact_labels[i]} (b = {impact_distances[i]:.1f} nm):")
        
        # Find peak energies
        peak_ret_idx = np.argmax(ptotal1[i, :])
        peak_qs_idx = np.argmax(ptotal2[i, :])
        
        peak_ret_energy = ene[peak_ret_idx]
        peak_qs_energy = ene[peak_qs_idx]
        
        peak_ret_intensity = ptotal1[i, peak_ret_idx]
        peak_qs_intensity = ptotal2[i, peak_qs_idx]
        
        print(f"  Retarded peak: {peak_ret_energy:.2f} eV, {peak_ret_intensity:.3e} eV⁻¹")
        print(f"  Quasistatic peak: {peak_qs_energy:.2f} eV, {peak_qs_intensity:.3e} eV⁻¹")
        
        # Surface vs bulk contribution analysis
        surf_contribution_ret = np.mean(psurf1[i, :] / (ptotal1[i, :] + 1e-15))
        surf_contribution_qs = np.mean(psurf2[i, :] / (ptotal2[i, :] + 1e-15))
        
        print(f"  Surface contribution (ret): {surf_contribution_ret*100:.1f}%")
        print(f"  Surface contribution (qs): {surf_contribution_qs*100:.1f}%")
    
    # Compare retarded vs quasistatic
    print(f"\nRetarded vs Quasistatic comparison:")
    for i in range(len(impact_labels)):
        rel_diff = np.mean(np.abs(ptotal1[i, :] - ptotal2[i, :]) / 
                          (ptotal2[i, :] + 1e-15)) * 100
        print(f"  {impact_labels[i]}: Mean relative difference = {rel_diff:.1f}%")


def plot_detailed_eels_analysis(ene, ptotal1, ptotal2, psurf1, psurf2, 
                               pbulk1, pbulk2, imp, diameter):
    """Detailed EELS analysis plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    impact_labels = ['Center (b=0)', 'Mid-radius (b=R/2)', 'Edge (b=R)']
    colors = ['red', 'blue', 'green']
    
    # Total loss probability for each impact parameter
    for i in range(3):
        axes[0, i].plot(ene, ptotal1[i, :], 'r-', label='Retarded', linewidth=2)
        axes[0, i].plot(ene, ptotal2[i, :], 'b--', label='Quasistatic', linewidth=2)
        axes[0, i].set_xlabel('Loss energy (eV)')
        axes[0, i].set_ylabel('Loss probability (eV⁻¹)')
        axes[0, i].set_title(impact_labels[i])
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
    
    # Surface vs bulk contributions
    for i in range(3):
        axes[1, i].plot(ene, psurf1[i, :], 'r-', label='Surface (ret)', linewidth=2)
        axes[1, i].plot(ene, pbulk1[i, :], 'r:', label='Bulk (ret)', linewidth=2)
        axes[1, i].plot(ene, psurf2[i, :], 'b--', label='Surface (qs)', linewidth=2)
        axes[1, i].plot(ene, pbulk2[i, :], 'b-.', label='Bulk (qs)', linewidth=2)
        axes[1, i].set_xlabel('Loss energy (eV)')
        axes[1, i].set_ylabel('Loss probability (eV⁻¹)')
        axes[1, i].set_title(f'Surface vs Bulk: {impact_labels[i]}')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_impact_parameter_dependence(ene, ptotal1, ptotal2, imp, diameter):
    """Plot impact parameter dependence at specific energies"""
    
    # Select a few representative energies
    energy_indices = [10, 20, 30]  # Low, mid, high energy
    selected_energies = ene[energy_indices]
    
    # Impact parameter distances
    impact_distances = np.array([0, 0.48 * diameter/2, 0.96 * diameter/2])
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    for i, idx in enumerate(energy_indices):
        plt.plot(impact_distances, ptotal1[:, idx], 'o-', 
                label=f'Retarded, {selected_energies[i]:.1f} eV', linewidth=2)
        plt.plot(impact_distances, ptotal2[:, idx], 's--', 
                label=f'Quasistatic, {selected_energies[i]:.1f} eV', linewidth=2)
    
    plt.xlabel('Impact parameter (nm)')
    plt.ylabel('Loss probability (eV⁻¹)')
    plt.title('Impact parameter dependence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Relative difference vs impact parameter
    for i, idx in enumerate(energy_indices):
        rel_diff = (ptotal1[:, idx] - ptotal2[:, idx]) / (ptotal2[:, idx] + 1e-15) * 100
        plt.plot(impact_distances, rel_diff, 'o-', 
                label=f'{selected_energies[i]:.1f} eV', linewidth=2)
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Impact parameter (nm)')
    plt.ylabel('Relative difference (%)')
    plt.title('(Retarded - Quasistatic)/Quasistatic × 100%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def explain_nanodisk_eels():
    """Explain nanodisk EELS physics"""
    
    print("\n=== Nanodisk EELS Physics ===")
    print("Impact parameter effects:")
    print("- Center (b=0): Excites all multipolar modes")
    print("- Edge (b=R): Mainly dipolar surface plasmons") 
    print("- Different modes have different spatial distributions")
    print("- Retardation effects become important for large particles")
    print("- Surface modes dominate over bulk modes for thin disks")


if __name__ == '__main__':
    ptotal1, ptotal2, psurf1, psurf2, pbulk1, pbulk2, ene, imp = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Silver nanodisk: 30 nm diameter, 5 nm height")
    print(f"Electron beam: 200 keV energy")
    print(f"Impact parameters: 0, R/2, R (0, 7.2, 14.4 nm)")
    print(f"Loss energy range: {ene[0]:.1f} - {ene[-1]:.1f} eV")
    print(f"Methods: Retarded vs Quasistatic BEM")
    
    # Detailed analysis
    plot_detailed_eels_analysis(ene, ptotal1, ptotal2, psurf1, psurf2, 
                               pbulk1, pbulk2, imp, 30)
    plot_impact_parameter_dependence(ene, ptotal1, ptotal2, imp, 30)
    explain_nanodisk_eels()