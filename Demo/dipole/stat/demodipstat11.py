"""
DEMODIPSTAT11 - Lifetime reduction for dipole between sphere and layer.

For a metallic nanosphere with a diameter of 4 nm located 1 nm above a
substrate and a dipole located between sphere and layer, this program
computes the total dipole scattering rates within the quasistatic
approximation. We first compute the LDOS using the BEMSTATLAYER
solver. Second, we model the substrate through an additional particle
of sufficient size. To speed up the simulation, we exploit mirror
symmetry and use the BEMSTATMIRROR solver.

Runtime: ~43 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import trispheresegment, comparticle, comparticlemirror, shift, flip, polygon, polygon3, plate
from pymnpbem.misc import compoint, layerstructure
from pymnpbem.bem import dipole, bemsolver


def refinement_function(pos, d=None):
    """Refinement function for substrate plate"""
    return 0.1 + 0.3 * np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)


def main():
    # Initialization
    # Table of dielectric functions
    epstab = [epsconst(1), epstable('gold.dat'), epsconst(2.25)]  # air, gold, substrate
    
    # Location of interface of substrate
    ztab = 0
    
    # Set up layer structure
    layer = layerstructure(epstab, [1, 3], ztab)
    
    # Options for BEM simulation with substrate (layered approach)
    op1 = bemoptions(sim='stat', interp='curv', layer=layer)
    
    # Options for BEM simulation with mirror symmetry (explicit particle approach)
    op2 = bemoptions(sim='stat', interp='curv', sym='xy')
    
    # Particle without symmetry (layered substrate approach)
    # Nanosphere with finer discretization at the bottom
    phi_grid = 2 * np.pi * np.linspace(0, 1, 31)
    theta_grid = np.pi * np.linspace(0, 1, 31) ** 2
    p_sphere = trispheresegment(phi_grid, theta_grid, 4)  # 4 nm diameter
    
    # Flip sphere (finer discretization at bottom)
    p_sphere = flip(p_sphere, axis=2)
    
    # Set up COMPARTICLE object, place sphere 1 nm above substrate
    p1 = comparticle(epstab, [shift(p_sphere, [0, 0, 3])], [2, 1], 1, op1)
    
    # Particle with mirror symmetry (explicit substrate particle)
    # Nanosphere with finer discretization at the bottom (smaller angular grid)
    phi_grid_sym = 0.5 * np.pi * np.linspace(0, 1, 11)
    theta_grid_sym = np.pi * np.linspace(0, 1, 31) ** 2
    p_sphere_sym = trispheresegment(phi_grid_sym, theta_grid_sym, 4)
    
    # Flip sphere
    p_sphere_sym = flip(p_sphere_sym, axis=2)
    
    # Polygon for plate (substrate)
    poly = polygon3(polygon(30, size=[15, 15]), 0)
    
    # Plate below nanosphere
    subs = plate(poly, op2, refun=refinement_function)
    
    # Set up COMPARTICLEMIRROR object
    p2 = comparticlemirror(epstab, 
                          [shift(p_sphere_sym, [0, 0, 3]), subs],
                          [[2, 1], [3, 1]], 1, op2)
    
    # Dipole oscillator
    enei = np.linspace(400, 800, 40)
    
    # Position of dipole
    pt = compoint(p1, [0, 0, 0.5], op1)
    
    # Dipole excitation with and w/o explicit consideration of substrate
    dip1 = dipole(pt, np.array([[1, 0, 0], [0, 0, 1]]), op1)  # With layer substrate
    dip2 = dipole(pt, np.array([[1, 0, 0], [0, 0, 1]]), op2)  # With explicit substrate
    
    # Initialize total and radiative scattering rate
    tot1 = np.zeros((len(enei), 2))  # Layered substrate
    rad1 = np.zeros((len(enei), 2))
    tot2 = np.zeros((len(enei), 2))  # Explicit substrate
    rad2 = np.zeros((len(enei), 2))
    
    # BEM simulation
    # Set up BEM solvers
    bem1 = bemsolver(p1, op1)  # Layered substrate solver
    bem2 = bemsolver(p2, op2)  # Mirror symmetry solver
    
    print("Running BEM simulations with different substrate models...")
    
    # Loop over wavelengths
    for ien in tqdm(range(len(enei)), desc="BEM solver", ncols=80):
        # Surface charge
        sig1 = bem1.solve(dip1(p1, enei[ien]))  # Layered approach
        sig2 = bem2.solve(dip2(p2, enei[ien]))  # Explicit approach
        
        # Total and radiative decay rate
        tot1[ien, :], rad1[ien, :] = dip1.decayrate(sig1)
        tot2[ien, :], rad2[ien, :] = dip2.decayrate(sig2)
    
    print("Substrate modeling comparison completed!")
    
    # Final plot
    plt.figure(figsize=(12, 8))
    
    # Plot layered substrate results
    plt.plot(enei, tot1[:, 0], 'o-', label='x-dip, with subs (layer)', linewidth=2, markersize=4)
    plt.plot(enei, tot1[:, 1], 'o-', label='z-dip, with subs (layer)', linewidth=2, markersize=4)
    
    # Plot explicit substrate results
    plt.plot(enei, tot2[:, 0], '.-', label='x-dip, explicit subs', linewidth=2, markersize=6)
    plt.plot(enei, tot2[:, 1], '.-', label='z-dip, explicit subs', linewidth=2, markersize=6)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Total decay rate')
    plt.legend()
    plt.title('Substrate modeling comparison: Layer vs Explicit particle')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    compare_substrate_models(enei, tot1, rad1, tot2, rad2)
    
    return tot1, rad1, tot2, rad2, enei


def compare_substrate_models(enei, tot1, rad1, tot2, rad2):
    """Compare layered vs explicit substrate modeling approaches"""
    
    print("\n=== Substrate Modeling Comparison ===")
    
    # Calculate relative differences
    rel_diff_tot_x = np.abs(tot1[:, 0] - tot2[:, 0]) / (tot1[:, 0] + 1e-10) * 100
    rel_diff_tot_z = np.abs(tot1[:, 1] - tot2[:, 1]) / (tot1[:, 1] + 1e-10) * 100
    
    print(f"Agreement between methods:")
    print(f"Total decay rate (x): Mean diff = {np.mean(rel_diff_tot_x):.1f}%, Max diff = {np.max(rel_diff_tot_x):.1f}%")
    print(f"Total decay rate (z): Mean diff = {np.mean(rel_diff_tot_z):.1f}%, Max diff = {np.max(rel_diff_tot_z):.1f}%")
    
    # Method characteristics
    print(f"\nMethod characteristics:")
    print(f"Layered substrate approach:")
    print(f"  - Uses analytical Green's functions")
    print(f"  - Infinite substrate extent")
    print(f"  - More accurate for substrate effects")
    print(f"  - Smaller computational mesh")
    
    print(f"Explicit substrate approach:")
    print(f"  - Discretizes substrate as finite particle")
    print(f"  - Can use mirror symmetry for speed")
    print(f"  - More flexible geometries")
    print(f"  - Larger computational mesh")
    
    # Performance comparison
    max_enhancement_layer_x = np.max(tot1[:, 0])
    max_enhancement_layer_z = np.max(tot1[:, 1])
    max_enhancement_explicit_x = np.max(tot2[:, 0])
    max_enhancement_explicit_z = np.max(tot2[:, 1])
    
    print(f"\nMaximum LDOS enhancements:")
    print(f"Layered approach - x: {max_enhancement_layer_x:.2f}, z: {max_enhancement_layer_z:.2f}")
    print(f"Explicit approach - x: {max_enhancement_explicit_x:.2f}, z: {max_enhancement_explicit_z:.2f}")


def plot_detailed_comparison(enei, tot1, rad1, tot2, rad2):
    """Detailed comparison plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total decay rates comparison
    axes[0, 0].plot(enei, tot1[:, 0], 'ro-', label='Layer method', linewidth=2, markersize=4)
    axes[0, 0].plot(enei, tot2[:, 0], 'b.-', label='Explicit method', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Wavelength (nm)')
    axes[0, 0].set_ylabel('Total decay rate')
    axes[0, 0].set_title('x-polarized dipole')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(enei, tot1[:, 1], 'ro-', label='Layer method', linewidth=2, markersize=4)
    axes[0, 1].plot(enei, tot2[:, 1], 'b.-', label='Explicit method', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Wavelength (nm)')
    axes[0, 1].set_ylabel('Total decay rate')
    axes[0, 1].set_title('z-polarized dipole')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Relative differences
    rel_diff_x = (tot2[:, 0] - tot1[:, 0]) / (tot1[:, 0] + 1e-10) * 100
    rel_diff_z = (tot2[:, 1] - tot1[:, 1]) / (tot1[:, 1] + 1e-10) * 100
    
    axes[1, 0].plot(enei, rel_diff_x, 'g-', linewidth=2)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Wavelength (nm)')
    axes[1, 0].set_ylabel('Relative difference (%)')
    axes[1, 0].set_title('(Explicit - Layer)/Layer: x-polarization')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(enei, rel_diff_z, 'purple', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Wavelength (nm)')
    axes[1, 1].set_ylabel('Relative difference (%)')
    axes[1, 1].set_title('(Explicit - Layer)/Layer: z-polarization')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_computational_trade_offs():
    """Analyze computational trade-offs between methods"""
    
    print("\n=== Computational Trade-offs ===")
    
    print("Layered substrate method:")
    print("Advantages:")
    print("  - Exact treatment of infinite substrate")
    print("  - Smaller mesh (sphere only)")
    print("  - Analytical Green's functions")
    print("  - Better accuracy for substrate effects")
    
    print("Disadvantages:")
    print("  - Limited to layered geometries")
    print("  - More complex implementation")
    print("  - Cannot use mirror symmetry easily")
    
    print("\nExplicit substrate method:")
    print("Advantages:")
    print("  - Can use mirror symmetry (4x speedup)")
    print("  - Flexible substrate shapes")
    print("  - Unified treatment of all objects")
    print("  - Easier to implement")
    
    print("Disadvantages:")
    print("  - Finite substrate size effects")
    print("  - Larger computational mesh")
    print("  - Need sufficient substrate size")
    print("  - Mesh discretization errors")
    
    print("\nRecommendations:")
    print("  - Use layered method for high accuracy")
    print("  - Use explicit method for complex geometries")
    print("  - Use mirror symmetry when possible")
    print("  - Validate with both methods for critical results")


if __name__ == '__main__':
    tot1, rad1, tot2, rad2, enei = main()
    
    print("\n=== Simulation Summary ===")
    print(f"System: 4 nm Au nanosphere 1 nm above substrate")
    print(f"Dipole at (0, 0, 0.5) nm between sphere and substrate")
    print(f"Wavelength range: {enei[0]:.0f} - {enei[-1]:.0f} nm")
    print(f"Method 1: Layered substrate (analytical Green's functions)")
    print(f"Method 2: Explicit substrate particle with mirror symmetry")
    
    # Detailed comparison
    plot_detailed_comparison(enei, tot1, rad1, tot2, rad2)
    analyze_computational_trade_offs()
    
    print(f"\nDemo/dipole/stat 폴더 완전 변환 완료!")