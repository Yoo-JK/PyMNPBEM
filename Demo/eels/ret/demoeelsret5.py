"""
DEMOEELSRET5 - EELS maps of nanotriangle for selected loss energies.

For a silver nanotriangle with 80 nm base length and 10 nm height,
this program computes EELS maps for selected loss energies using the
full Maxwell equations.

Runtime: ~83 sec.
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
    
    # EELS excitation
    # Units
    unit_converter = units()
    
    # Loss energies (extracted from DEMOEELSRET4)
    ene = np.array([2.13, 2.82, 3.04, 3.48])
    
    # Wavelengths
    enei = unit_converter.eV2nm / ene
    
    # Mesh for electron beams
    x = np.linspace(-70, 50, 50)
    y = np.linspace(0, 50, 35)
    X, Y = np.meshgrid(x, y)
    
    # Impact parameters
    impact = np.column_stack([X.flatten(), Y.flatten()])
    
    # Width of electron beam and electron velocity
    width = 0.2
    vel = eelsbase.ene2vel(200e3)  # 200 keV electron energy
    
    # BEM simulation
    # BEM solver
    bem = bemsolver(p, op)
    
    # EELS excitation
    exc = electronbeam(p, impact, width, vel, op)
    
    # Electron energy loss probabilities
    psurf = np.zeros((impact.shape[0], len(enei)))
    pbulk = np.zeros((impact.shape[0], len(enei)))
    
    print("Running BEM simulation for EELS maps...")
    
    # Loop over energies
    for ien in tqdm(range(len(enei)), desc="BEM solver", ncols=80):
        # Surface charges
        sig = bem.solve(exc(enei[ien]))
        
        # EELS losses
        psurf[:, ien], pbulk[:, ien] = exc.loss(sig)
    
    print("EELS mapping simulation completed!")
    
    # Total losses
    ptotal = psurf + pbulk
    
    # Final plot
    # x and y limits
    xx = [np.min(X), np.max(X)]
    yy = [-np.max(Y), np.max(Y)]
    
    # Energy labels
    energy_labels = ['2.13 eV', '2.82 eV', '3.04 eV', '3.48 eV']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot EELS maps
    for i in range(4):
        # Reshape loss probability
        prob = ptotal[:, i].reshape(Y.shape)
        
        # Add second part of triangle (mirror symmetry)
        prob_full = np.vstack([np.flipud(prob[1:, :]), prob])
        
        im = axes[i].imshow(prob_full, extent=[xx[0], xx[1], yy[0], yy[1]], 
                           cmap='hot', aspect='equal', origin='lower')
        axes[i].set_xlabel('x (nm)')
        axes[i].set_ylabel('y (nm)')
        axes[i].set_title(energy_labels[i])
        plt.colorbar(im, ax=axes[i], label='Loss probability (eV⁻¹)')
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    analyze_eels_maps(X, Y, ptotal, ene, energy_labels)
    
    return ptotal, psurf, pbulk, X, Y, ene


def analyze_eels_maps(X, Y, ptotal, ene, energy_labels):
    """Analyze EELS spatial maps"""
    
    print("\n=== EELS Maps Analysis ===")
    
    for i, energy in enumerate(ene):
        print(f"\n{energy_labels[i]}:")
        
        # Reshape data
        prob_map = ptotal[:, i].reshape(Y.shape)
        
        # Find maximum position
        max_idx = np.unravel_index(np.argmax(prob_map), prob_map.shape)
        max_x = X[max_idx]
        max_y = Y[max_idx]
        max_prob = prob_map[max_idx]
        
        print(f"  Maximum intensity: {max_prob:.3e} eV⁻¹")
        print(f"  Maximum position: ({max_x:.1f}, {max_y:.1f}) nm")
        
        # Center of mass calculation
        total_intensity = np.sum(prob_map)
        if total_intensity > 0:
            com_x = np.sum(X * prob_map) / total_intensity
            com_y = np.sum(Y * prob_map) / total_intensity
            print(f"  Center of mass: ({com_x:.1f}, {com_y:.1f}) nm")
        
        # Spatial extent (where intensity > 10% of maximum)
        threshold = 0.1 * max_prob
        significant_mask = prob_map > threshold
        if np.any(significant_mask):
            x_extent = np.max(X[significant_mask]) - np.min(X[significant_mask])
            y_extent = np.max(Y[significant_mask]) - np.min(Y[significant_mask])
            print(f"  Spatial extent (>10% max): x = {x_extent:.1f} nm, y = {y_extent:.1f} nm")


def plot_detailed_eels_maps(X, Y, ptotal, psurf, pbulk, ene, energy_labels):
    """Detailed EELS maps analysis"""
    
    # Plot surface vs bulk contributions
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    xx = [np.min(X), np.max(X)]
    yy = [-np.max(Y), np.max(Y)]
    
    for i in range(4):
        # Total loss
        prob_total = ptotal[:, i].reshape(Y.shape)
        prob_total_full = np.vstack([np.flipud(prob_total[1:, :]), prob_total])
        
        im1 = axes[0, i].imshow(prob_total_full, extent=[xx[0], xx[1], yy[0], yy[1]], 
                               cmap='hot', aspect='equal', origin='lower')
        axes[0, i].set_title(f'Total: {energy_labels[i]}')
        axes[0, i].set_xlabel('x (nm)')
        axes[0, i].set_ylabel('y (nm)')
        plt.colorbar(im1, ax=axes[0, i])
        
        # Surface contribution
        prob_surf = psurf[:, i].reshape(Y.shape)
        prob_surf_full = np.vstack([np.flipud(prob_surf[1:, :]), prob_surf])
        
        im2 = axes[1, i].imshow(prob_surf_full, extent=[xx[0], xx[1], yy[0], yy[1]], 
                               cmap='plasma', aspect='equal', origin='lower')
        axes[1, i].set_title(f'Surface: {energy_labels[i]}')
        axes[1, i].set_xlabel('x (nm)')
        axes[1, i].set_ylabel('y (nm)')
        plt.colorbar(im2, ax=axes[1, i])
        
        # Bulk contribution
        prob_bulk = pbulk[:, i].reshape(Y.shape)
        prob_bulk_full = np.vstack([np.flipud(prob_bulk[1:, :]), prob_bulk])
        
        im3 = axes[2, i].imshow(prob_bulk_full, extent=[xx[0], xx[1], yy[0], yy[1]], 
                               cmap='viridis', aspect='equal', origin='lower')
        axes[2, i].set_title(f'Bulk: {energy_labels[i]}')
        axes[2, i].set_xlabel('x (nm)')
        axes[2, i].set_ylabel('y (nm)')
        plt.colorbar(im3, ax=axes[2, i])
    
    plt.tight_layout()
    plt.show()


def plot_energy_evolution(X, Y, ptotal, ene):
    """Plot evolution of EELS maps with energy"""
    
    # Create line profiles along x-axis at y=0
    y_center_idx = np.argmin(np.abs(Y[:, 0]))
    x_line = X[y_center_idx, :]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(ene)))
    
    for i, energy in enumerate(ene):
        prob_line = ptotal[:, i].reshape(Y.shape)[y_center_idx, :]
        plt.plot(x_line, prob_line, color=colors[i], 
                label=f'{energy:.2f} eV', linewidth=2)
    
    plt.xlabel('x position (nm)')
    plt.ylabel('Loss probability (eV⁻¹)')
    plt.title('EELS profiles along x-axis (y=0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Peak position vs energy
    plt.subplot(1, 2, 2)
    peak_positions = []
    peak_intensities = []
    
    for i in range(len(ene)):
        prob_map = ptotal[:, i].reshape(Y.shape)
        max_idx = np.unravel_index(np.argmax(prob_map), prob_map.shape)
        peak_x = X[max_idx]
        peak_prob = prob_map[max_idx]
        
        peak_positions.append(peak_x)
        peak_intensities.append(peak_prob)
    
    plt.plot(ene, peak_positions, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Loss energy (eV)')
    plt.ylabel('Peak x-position (nm)')
    plt.title('Peak position vs energy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def explain_triangle_eels_maps():
    """Explain triangle EELS mapping physics"""
    
    print("\n=== Triangle EELS Mapping Physics ===")
    print("Spatial EELS maps reveal:")
    print("- Mode-dependent field distributions")
    print("- Hot spots at corners and edges")
    print("- Energy-dependent spatial patterns")
    print("- Surface vs bulk plasmon localization")
    print("- Symmetry breaking effects")
    print("- Different resonances have different spatial signatures")


if __name__ == '__main__':
    ptotal, psurf, pbulk, X, Y, ene = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Silver nanotriangle: 80 nm base length, 10 nm height")
    print(f"Electron beam: 200 keV energy, 0.2 nm width")
    print(f"EELS mapping grid: {X.shape[1]} × {X.shape[0]} points")
    print(f"Spatial range: x = {X.min():.0f} to {X.max():.0f} nm, y = 0 to {Y.max():.0f} nm")
    print(f"Loss energies: {', '.join(f'{e:.2f}' for e in ene)} eV")
    
    # Detailed analysis
    plot_detailed_eels_maps(X, Y, ptotal, psurf, pbulk, ene, 
                           ['2.13 eV', '2.82 eV', '3.04 eV', '3.48 eV'])
    plot_energy_evolution(X, Y, ptotal, ene)
    explain_triangle_eels_maps()