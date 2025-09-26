"""
DEMODIPRET5 - Photonic LDOS for nanotriangle (contd).

For a silver nanotriangle with 80 nm base length and 10 nm height,
this program computes the photonic LDOS in a plane 10 nm below the
nanoparticle and for transition dipole energies tuned to a few
selected plasmon resonances (extracted from DEMODIPRET4) using the
full Maxwell equations.

Runtime: ~2 min.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import polygon, tripolygon, comparticle, particle, shift, fvgrid
from pymnpbem.mesh2d import edgeprofile
from pymnpbem.misc import compoint, units
from pymnpbem.bem import dipole, bemsolver


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
    
    # Set up COMPARTICLE objects
    p = comparticle(epstab, [tripolygon(poly, edge)], [2, 1], 1, op)
    
    # Dipole oscillator
    # Transition energies (eV), extracted from DEMODIPRET4
    ene = np.array([2.14, 2.84, 3.04, 3.23, 3.49, 3.7])
    
    # Transform to wavelengths
    unit_converter = units()
    enei = unit_converter.eV2nm / ene
    
    # Positions for grid
    x = np.linspace(-70, 50, 31)
    y = np.linspace(-60, 60, 31)
    
    # Make grid
    verts, faces = fvgrid(x, y)
    
    # Make particle (plane 15 nm below the triangle)
    plate = shift(particle(verts, faces), [0, 0, -15])
    
    # Compoint
    pt = compoint(p, plate.verts)
    
    # Dipole excitation
    dip = dipole(pt, op)
    
    # Initialize total and radiative scattering rate
    tot = np.zeros((len(x) * len(y), len(enei), 3))
    rad = np.zeros((len(x) * len(y), len(enei), 3))
    
    # BEM simulation
    # Set up BEM solver
    bem = bemsolver(p, op)
    
    print("Running BEM simulation for LDOS mapping...")
    
    # Loop over wavelengths
    for ien in tqdm(range(len(enei)), desc="BEM solver", ncols=80):
        # Surface charges
        sig = bem.solve(dip(p, enei[ien]))
        
        # Total and radiative decay rate
        tot[:, ien, :], rad[:, ien, :] = dip.decayrate(sig)
    
    print("BEM simulation completed!")
    
    # Reshape data for plotting
    X, Y = np.meshgrid(x, y)
    ldos_maps = np.zeros((len(y), len(x), len(enei)))
    
    # Average over polarizations and reshape to grid
    ldos_avg = np.sum(tot, axis=2) / 3  # Average over polarizations
    
    for ien in range(len(enei)):
        ldos_maps[:, :, ien] = ldos_avg[:, ien].reshape(len(y), len(x))
    
    # Plot results
    plot_ldos_maps(X, Y, ldos_maps, ene, p)
    plot_3d_visualization(X, Y, ldos_maps, ene)
    
    return tot, rad, ene, enei, X, Y, ldos_maps


def plot_ldos_maps(X, Y, ldos_maps, ene, particle=None):
    """Plot 2D LDOS maps for each energy"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, energy in enumerate(ene):
        im = axes[i].contourf(X, Y, ldos_maps[:, :, i], levels=20, cmap='hot')
        axes[i].set_title(f'LDOS at {energy:.2f} eV ({unit_converter.eV2nm/energy:.0f} nm)')
        axes[i].set_xlabel('x (nm)')
        axes[i].set_ylabel('y (nm)')
        axes[i].set_aspect('equal')
        
        # Add triangle outline if available
        if particle is not None:
            # This would need to be implemented based on particle structure
            # axes[i].plot(triangle_x, triangle_y, 'w-', linewidth=2)
            pass
        
        plt.colorbar(im, ax=axes[i], label='LDOS enhancement')
    
    plt.tight_layout()
    plt.show()


def plot_3d_visualization(X, Y, ldos_maps, ene):
    """Create 3D surface plots for selected energies"""
    
    # Select a few interesting energies for 3D visualization
    selected_indices = [0, 2, 4]  # First, middle, and one near the end
    
    fig = plt.figure(figsize=(18, 6))
    
    for i, idx in enumerate(selected_indices):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        surf = ax.plot_surface(X, Y, ldos_maps[:, :, idx], 
                              cmap='hot', alpha=0.8, linewidth=0, antialiased=True)
        
        ax.set_title(f'LDOS at {ene[idx]:.2f} eV')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_zlabel('LDOS enhancement')
        
        fig.colorbar(surf, ax=ax, shrink=0.5)
    
    plt.tight_layout()
    plt.show()


def analyze_spatial_distribution(X, Y, ldos_maps, ene):
    """Analyze spatial characteristics of LDOS maps"""
    
    print("\n=== Spatial LDOS Analysis ===")
    
    for i, energy in enumerate(ene):
        ldos_map = ldos_maps[:, :, i]
        
        # Find maximum and its position
        max_idx = np.unravel_index(np.argmax(ldos_map), ldos_map.shape)
        max_value = ldos_map[max_idx]
        max_x = X[max_idx]
        max_y = Y[max_idx]
        
        # Calculate mean and standard deviation
        mean_ldos = np.mean(ldos_map)
        std_ldos = np.std(ldos_map)
        
        print(f"Energy {energy:.2f} eV:")
        print(f"  Maximum LDOS: {max_value:.2f} at ({max_x:.1f}, {max_y:.1f}) nm")
        print(f"  Mean LDOS: {mean_ldos:.2f} ± {std_ldos:.2f}")
        print(f"  Enhancement factor: {max_value/mean_ldos:.1f}x")
        print()


def plot_particle_with_ldos(p, plate, ldos_data):
    """Plot particle geometry with LDOS data"""
    
    fig = plt.figure(figsize=(15, 10))
    
    # Plot particle geometry
    ax1 = fig.add_subplot(121, projection='3d')
    # This would need particle plotting capabilities
    # plot(p, FaceAlpha=0.5)  # MATLAB equivalent
    ax1.set_title('Silver Nanotriangle')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    ax1.set_zlabel('z (nm)')
    
    # Plot LDOS data on the plane
    ax2 = fig.add_subplot(122, projection='3d')
    # plot(plate, ldos_data)  # MATLAB equivalent
    ax2.set_title('LDOS Distribution')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    ax2.set_zlabel('z (nm)')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    tot, rad, ene, enei, X, Y, ldos_maps = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Selected resonant energies: {', '.join(f'{e:.2f}' for e in ene)} eV")
    print(f"Corresponding wavelengths: {', '.join(f'{w:.0f}' for w in enei)} nm")
    print(f"Grid size: {X.shape[0]} x {X.shape[1]}")
    print(f"Spatial range: x = [{X.min():.0f}, {X.max():.0f}] nm, y = [{Y.min():.0f}, {Y.max():.0f}] nm")
    print(f"Measurement plane: 15 nm below triangle")
    
    # Analyze spatial distribution
    analyze_spatial_distribution(X, Y, ldos_maps, ene)
    
    print("Note: 3D particle visualization requires implementation of particle plotting methods.")