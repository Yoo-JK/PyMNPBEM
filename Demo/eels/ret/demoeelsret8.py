"""
DEMOEELSRET8 - EELS maps for silver nanotriangle on membrane.

For a silver nanotriangle with 80 nm base length and 10 nm height,
which is located on a 15 nm thin membrane, this program computes the
EELS maps for selected loss energies using the full Maxwell equations.

Runtime: ~6 min.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import polygon, polygon3, tripolygon, comparticle, plate, particle, fvgrid, shift, flipfaces, select
from pymnpbem.mesh2d import edgeprofile
from pymnpbem.misc import units, eelsbase
from pymnpbem.bem import bemsolver, electronbeam


def main():
    # Initialization
    # Options for BEM simulation
    op = bemoptions(sim='ret', interp='curv', eels_refine=2)
    
    # Table of dielectric functions
    epstab = [epsconst(1), epstable('silver.dat'), epsconst(4)]  # air, silver, membrane
    
    # Dimensions of particle
    length = [80, 80 * 2 / np.sqrt(3)]
    
    # Polygon
    poly = polygon(3, size=length).round()
    
    # Edge profile
    edge = edgeprofile(10, 11, mode='01', min=0)
    
    # Mesh data structure
    hdata = {'hmax': 8}
    
    # Extrude polygon
    p, poly = tripolygon(poly, edge, hdata=hdata)
    
    print("Creating membrane geometry with nanotriangle...")
    
    # Split into upper and lower part
    pup, plo = select(p, carfun=lambda x, y, z: z > 1e-3)
    
    # Polygon for plate (membrane)
    poly2 = polygon3(polygon(4, size=[150, 150]), 0)
    
    # Upper plate
    up = plate([poly2, poly.set_z(0)], dir=-1)
    
    # Lower plate
    x = 150 * np.linspace(-0.5, 0.5, 21)
    verts, faces = fvgrid(x, x)
    lo = flipfaces(shift(particle(verts, faces), [0, 0, -15]))
    
    # Make complete particle system
    p = comparticle(epstab, [pup, plo, up, lo],
                   [[2, 1], [2, 3], [1, 3], [3, 1]], [1, 2], op)
    
    # EELS excitation
    # Loss energies (extracted from DEMOEELSRET7)
    ene = np.array([1.60, 2.15, 2.38, 2.90])
    
    # Convert energies to nm
    unit_converter = units()
    enei = unit_converter.eV2nm / ene
    
    # Mesh for electron beams
    x = np.linspace(-70, 50, 50)
    y = np.linspace(0, 50, 35)
    X, Y = np.meshgrid(x, y)
    
    # Impact parameters
    impact = np.column_stack([X.flatten(), Y.flatten()])
    
    # Width of electron beam and electron velocity
    width = 0.2
    vel = eelsbase.ene2vel(200e3)
    
    # BEM simulation
    bem = bemsolver(p, op)
    exc = electronbeam(p, impact, width, vel, op)
    
    # Electron energy loss probabilities
    psurf = np.zeros((impact.shape[0], len(enei)))
    pbulk = np.zeros((impact.shape[0], len(enei)))
    
    print("Running EELS mapping simulation on membrane...")
    print("This may take several minutes...")
    
    # Loop over energies
    for ien in tqdm(range(len(ene)), desc="BEM solver", ncols=80):
        # Surface charges
        sig = bem.solve(exc(enei[ien]))
        
        # EELS losses
        psurf[:, ien], pbulk[:, ien] = exc.loss(sig)
    
    print("EELS mapping on membrane completed!")
    
    # Total losses
    ptotal = psurf + pbulk
    
    # x and y limits
    xx = [np.min(X), np.max(X)]
    yy = [-np.max(Y), np.max(Y)]
    
    # Energy labels
    energy_labels = ['1.60 eV', '2.15 eV', '2.38 eV', '2.90 eV']
    
    # Final plot
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
    
    plt.suptitle('EELS Maps: Silver Nanotriangle on Membrane')
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    analyze_membrane_eels_maps(X, Y, ptotal, psurf, pbulk, ene, energy_labels)
    
    return ptotal, psurf, pbulk, X, Y, ene


def analyze_membrane_eels_maps(X, Y, ptotal, psurf, pbulk, ene, energy_labels):
    """Analyze EELS maps on membrane substrate"""
    
    print("\n=== Membrane EELS Maps Analysis ===")
    
    for i, energy in enumerate(ene):
        print(f"\n{energy_labels[i]}:")
        
        # Reshape data
        prob_map = ptotal[:, i].reshape(Y.shape)
        surf_map = psurf[:, i].reshape(Y.shape)
        bulk_map = pbulk[:, i].reshape(Y.shape)
        
        # Maximum position and intensity
        max_idx = np.unravel_index(np.argmax(prob_map), prob_map.shape)
        max_x = X[max_idx]
        max_y = Y[max_idx]
        max_prob = prob_map[max_idx]
        
        print(f"  Maximum intensity: {max_prob:.3e} eV⁻¹")
        print(f"  Maximum position: ({max_x:.1f}, {max_y:.1f}) nm")
        
        # Surface vs bulk contributions at maximum
        surf_at_max = surf_map[max_idx]
        bulk_at_max = bulk_map[max_idx]
        
        if max_prob > 0:
            surf_fraction = surf_at_max / max_prob
            bulk_fraction = bulk_at_max / max_prob
            print(f"  Surface/bulk ratio at max: {surf_fraction:.2f}/{bulk_fraction:.2f}")
        
        # Spatial distribution characteristics
        total_intensity = np.sum(prob_map)
        if total_intensity > 0:
            com_x = np.sum(X * prob_map) / total_intensity
            com_y = np.sum(Y * prob_map) / total_intensity
            print(f"  Center of mass: ({com_x:.1f}, {com_y:.1f}) nm")


def plot_membrane_vs_surface_contributions(X, Y, ptotal, psurf, pbulk, ene, energy_labels):
    """Plot surface vs bulk contributions on membrane"""
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    xx = [np.min(X), np.max(X)]
    yy = [-np.max(Y), np.max(Y)]
    
    for i in range(4):
        # Total
        prob_total = ptotal[:, i].reshape(Y.shape)
        prob_total_full = np.vstack([np.flipud(prob_total[1:, :]), prob_total])
        
        im1 = axes[0, i].imshow(prob_total_full, extent=[xx[0], xx[1], yy[0], yy[1]], 
                               cmap='hot', aspect='equal', origin='lower')
        axes[0, i].set_title(f'Total: {energy_labels[i]}')
        axes[0, i].set_xlabel('x (nm)')
        axes[0, i].set_ylabel('y (nm)')
        plt.colorbar(im1, ax=axes[0, i])
        
        # Surface
        prob_surf = psurf[:, i].reshape(Y.shape)
        prob_surf_full = np.vstack([np.flipud(prob_surf[1:, :]), prob_surf])
        
        im2 = axes[1, i].imshow(prob_surf_full, extent=[xx[0], xx[1], yy[0], yy[1]], 
                               cmap='plasma', aspect='equal', origin='lower')
        axes[1, i].set_title(f'Surface: {energy_labels[i]}')
        axes[1, i].set_xlabel('x (nm)')
        axes[1, i].set_ylabel('y (nm)')
        plt.colorbar(im2, ax=axes[1, i])
        
        # Bulk
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


def compare_membrane_effects(X, Y, ptotal, ene):
    """Compare membrane effects across different energies"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Energy evolution at triangle center
    center_x_idx = np.argmin(np.abs(X[0, :]))
    center_y_idx = np.argmin(np.abs(Y[:, 0]))
    center_idx = center_y_idx * X.shape[1] + center_x_idx
    
    center_intensities = [ptotal[center_idx, i] for i in range(len(ene))]
    
    axes[0].plot(ene, center_intensities, 'ro-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Loss energy (eV)')
    axes[0].set_ylabel('Loss probability at center (eV⁻¹)')
    axes[0].set_title('Energy Evolution at Triangle Center')
    axes[0].grid(True, alpha=0.3)
    
    # Peak position vs energy
    peak_x_positions = []
    peak_y_positions = []
    peak_intensities = []
    
    for i in range(len(ene)):
        prob_map = ptotal[:, i].reshape(Y.shape)
        max_idx = np.unravel_index(np.argmax(prob_map), prob_map.shape)
        peak_x = X[max_idx]
        peak_y = Y[max_idx]
        peak_intensity = prob_map[max_idx]
        
        peak_x_positions.append(peak_x)
        peak_y_positions.append(peak_y)
        peak_intensities.append(peak_intensity)
    
    # Scatter plot of peak positions
    scatter = axes[1].scatter(peak_x_positions, peak_y_positions, 
                             c=ene, s=[100*p/max(peak_intensities) for p in peak_intensities], 
                             cmap='viridis', alpha=0.7)
    
    axes[1].set_xlabel('Peak x-position (nm)')
    axes[1].set_ylabel('Peak y-position (nm)')
    axes[1].set_title('Peak Position vs Energy (size ∝ intensity)')
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label('Energy (eV)')
    axes[1].grid(True, alpha=0.3)
    
    # Add energy labels
    for i, (x, y, energy) in enumerate(zip(peak_x_positions, peak_y_positions, ene)):
        axes[1].annotate(f'{energy:.2f}', (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def explain_membrane_mapping():
    """Explain membrane EELS mapping physics"""
    
    print("\n=== Membrane EELS Mapping Physics ===")
    print("Membrane effects on spatial EELS maps:")
    print("- Modified field confinement vs bulk substrate")
    print("- Interface plasmons at membrane boundaries")
    print("- Finite thickness creates standing wave patterns")
    print("- More realistic experimental geometry")
    print("- Enhanced spatial resolution near interfaces")
    print("- Energy-dependent penetration into substrate")


if __name__ == '__main__':
    ptotal, psurf, pbulk, X, Y, ene = main()
    
    print("\n=== Final Simulation Summary ===")
    print(f"Silver nanotriangle: 80 nm base, 10 nm height")
    print(f"Membrane: 15 nm thick, ε = 4")
    print(f"EELS mapping: {X.shape[1]} × {X.shape[0]} grid points")
    print(f"Spatial range: x = {X.min():.0f} to {X.max():.0f} nm")
    print(f"Energy points: {', '.join(f'{e:.2f}' for e in ene)} eV")
    print(f"Complex geometry: Realistic TEM sample conditions")
    
    # Detailed analysis
    plot_membrane_vs_surface_contributions(X, Y, ptotal, psurf, pbulk, ene, 
                                          ['1.60 eV', '2.15 eV', '2.38 eV', '2.90 eV'])
    compare_membrane_effects(X, Y, ptotal, ene)
    explain_membrane_mapping()
    
    print(f"\nDemo/eels/ret 폴더의 모든 8개 예제가 완전히 변환되었습니다!")
    print(f"전자 에너지 손실 분광법의 모든 지연 효과 계산 예제가 완성되었습니다.")