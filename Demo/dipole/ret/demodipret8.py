"""
DEMODIPRET8 - Lifetime reduction for dipole between sphere and layer.

For a metallic nanosphere with a diameter of 4 nm located 1 nm above a
substrate and a dipole located between sphere and layer, this program
computes the total dipole scattering rates using the full Maxwell
equations, and compares the results with those obtained within the
quasistatic approximation.

Runtime: ~24 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import trispheresegment, comparticle, shift, flip
from pymnpbem.misc import compoint, layerstructure, tabspace, compgreentablayer
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
    op1 = bemoptions(sim='stat', interp='curv', layer=layer)  # Quasistatic
    op2 = bemoptions(sim='ret', interp='curv', layer=layer)   # Retarded
    
    # Nanosphere with finer discretization at the bottom
    phi_grid = 2 * np.pi * np.linspace(0, 1, 31)
    theta_grid = np.pi * np.linspace(0, 1, 31) ** 2
    p = trispheresegment(phi_grid, theta_grid, 4)  # 4 nm diameter
    
    # Flip sphere (finer discretization at bottom)
    p = flip(p, axis=2)  # flip along z-axis
    
    # Place nanosphere 1 nm above substrate
    min_z = np.min(p.pos[:, 2])
    p = shift(p, [0, 0, -min_z + 1 + ztab])
    
    # Set up COMPARTICLE object
    p = comparticle(epstab, [p], [2, 1], 1, op1)  # gold sphere in air
    
    # Dipole oscillator
    enei = 550  # wavelength in nm
    
    # Positions of dipole (between sphere and substrate)
    x = np.linspace(0, 5, 51).reshape(-1, 1)
    
    # Compoint - dipoles at z = 0.5 nm (between sphere and substrate)
    positions = np.column_stack([x.flatten(), np.zeros_like(x.flatten()), 
                                np.full_like(x.flatten(), 0.5)])
    pt = compoint(p, positions, op1)
    
    print("Setting up Green's function table for layered medium...")
    
    # Tabulated Green functions
    # For retarded simulation, set up table for reflected Green function
    # This is the most computationally expensive part
    
    # Automatic grid for tabulation (using small NZ for speed)
    tab = tabspace(layer, [p, pt], nz=5)
    
    # Green function table
    greentab = compgreentablayer(layer, tab)
    
    # Precompute Green function table
    greentab = greentab.set(enei, op2, waitbar=False)
    
    # Add Green table to retarded options
    op2.greentab = greentab
    
    print("Green's function table computed successfully!")
    
    # Dipole excitation
    dip1 = dipole(pt, np.array([[1, 0, 0], [0, 0, 1]]), op1)  # Quasistatic
    dip2 = dipole(pt, np.array([[1, 0, 0], [0, 0, 1]]), op2)  # Retarded
    
    print("Running BEM simulations...")
    
    # BEM simulation
    # Set up BEM solvers
    bem1 = bemsolver(p, op1)  # Quasistatic
    bem2 = bemsolver(p, op2)  # Retarded
    
    # Surface charge
    print("  Quasistatic calculation...")
    sig1 = bem1.solve(dip1(p, enei))
    
    print("  Retarded calculation...")
    sig2 = bem2.solve(dip2(p, enei))
    
    # Total and radiative decay rate
    tot1, rad1 = dip1.decayrate(sig1)
    tot2, rad2 = dip2.decayrate(sig2)
    
    print("BEM simulations completed!")
    
    # Final plot
    plt.figure(figsize=(12, 8))
    
    # Plot quasistatic results
    plt.plot(x.flatten(), tot1[:, 0], 'o-', label='x-dip (qs)', markersize=4)
    plt.plot(x.flatten(), tot1[:, 1], 'o-', label='z-dip (qs)', markersize=4)
    
    # Plot retarded results
    plt.plot(x.flatten(), tot2[:, 0], '.-', label='x-dip (ret)', markersize=6)
    plt.plot(x.flatten(), tot2[:, 1], '.-', label='z-dip (ret)', markersize=6)
    
    plt.xlabel('Position (nm)')
    plt.ylabel('Total decay rate')
    plt.legend()
    plt.title('Dipole decay rates: 4nm Au sphere 1nm above substrate (λ = 550nm)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    plot_detailed_analysis(x, tot1, rad1, tot2, rad2)
    
    return tot1, rad1, tot2, rad2, x


def plot_detailed_analysis(x, tot1, rad1, tot2, rad2):
    """Detailed analysis of quasistatic vs retarded effects"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total decay rates comparison
    axes[0, 0].plot(x.flatten(), tot1[:, 0], 'ro-', label='x-pol (qs)', markersize=4)
    axes[0, 0].plot(x.flatten(), tot2[:, 0], 'r.-', label='x-pol (ret)', markersize=6)
    axes[0, 0].set_xlabel('Position (nm)')
    axes[0, 0].set_ylabel('Total decay rate')
    axes[0, 0].set_title('x-polarized dipole')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(x.flatten(), tot1[:, 1], 'bo-', label='z-pol (qs)', markersize=4)
    axes[0, 1].plot(x.flatten(), tot2[:, 1], 'b.-', label='z-pol (ret)', markersize=6)
    axes[0, 1].set_xlabel('Position (nm)')
    axes[0, 1].set_ylabel('Total decay rate')
    axes[0, 1].set_title('z-polarized dipole')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Radiative decay rates
    axes[1, 0].plot(x.flatten(), rad1[:, 0], 'ro-', label='x-pol (qs)', markersize=4)
    axes[1, 0].plot(x.flatten(), rad2[:, 0], 'r.-', label='x-pol (ret)', markersize=6)
    axes[1, 0].set_xlabel('Position (nm)')
    axes[1, 0].set_ylabel('Radiative decay rate')
    axes[1, 0].set_title('Radiative: x-polarized')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(x.flatten(), rad1[:, 1], 'bo-', label='z-pol (qs)', markersize=4)
    axes[1, 1].plot(x.flatten(), rad2[:, 1], 'b.-', label='z-pol (ret)', markersize=6)
    axes[1, 1].set_xlabel('Position (nm)')
    axes[1, 1].set_ylabel('Radiative decay rate')
    axes[1, 1].set_title('Radiative: z-polarized')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Relative differences plot
    plt.figure(figsize=(12, 6))
    
    # Calculate relative differences (ret - qs) / qs * 100
    rel_diff_tot_x = (tot2[:, 0] - tot1[:, 0]) / tot1[:, 0] * 100
    rel_diff_tot_z = (tot2[:, 1] - tot1[:, 1]) / tot1[:, 1] * 100
    rel_diff_rad_x = (rad2[:, 0] - rad1[:, 0]) / rad1[:, 0] * 100
    rel_diff_rad_z = (rad2[:, 1] - rad1[:, 1]) / rad1[:, 1] * 100
    
    plt.subplot(1, 2, 1)
    plt.plot(x.flatten(), rel_diff_tot_x, 'r-', label='x-polarization', linewidth=2)
    plt.plot(x.flatten(), rel_diff_tot_z, 'b-', label='z-polarization', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Position (nm)')
    plt.ylabel('Relative difference (%)')
    plt.title('Total decay rate: (Retarded - Quasistatic)/Quasistatic')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(x.flatten(), rel_diff_rad_x, 'r-', label='x-polarization', linewidth=2)
    plt.plot(x.flatten(), rel_diff_rad_z, 'b-', label='z-polarization', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Position (nm)')
    plt.ylabel('Relative difference (%)')
    plt.title('Radiative decay rate: (Retarded - Quasistatic)/Quasistatic')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_substrate_effects(x, tot1, rad1, tot2, rad2):
    """Analyze substrate and retardation effects"""
    
    print("\n=== Substrate and Retardation Effects Analysis ===")
    
    # Find positions of maximum enhancement
    for pol, label in enumerate(['x', 'z']):
        print(f"\n{label.upper()}-polarization:")
        
        # Maximum decay rates
        max_tot1_idx = np.argmax(tot1[:, pol])
        max_tot2_idx = np.argmax(tot2[:, pol])
        
        print(f"  Quasistatic maximum: {tot1[max_tot1_idx, pol]:.2f} at x = {x[max_tot1_idx, 0]:.2f} nm")
        print(f"  Retarded maximum: {tot2[max_tot2_idx, pol]:.2f} at x = {x[max_tot2_idx, 0]:.2f} nm")
        
        # Enhancement at contact (x = 0)
        contact_tot1 = tot1[0, pol]
        contact_tot2 = tot2[0, pol]
        contact_diff = (contact_tot2 - contact_tot1) / contact_tot1 * 100
        
        print(f"  At contact (x = 0): qs = {contact_tot1:.2f}, ret = {contact_tot2:.2f}")
        print(f"  Retardation effect at contact: {contact_diff:+.1f}%")
        
        # Far-field behavior (x = max)
        far_tot1 = tot1[-1, pol]
        far_tot2 = tot2[-1, pol]
        far_diff = (far_tot2 - far_tot1) / far_tot1 * 100
        
        print(f"  At x = {x[-1, 0]:.1f} nm: qs = {far_tot1:.2f}, ret = {far_tot2:.2f}")
        print(f"  Retardation effect far field: {far_diff:+.1f}%")


if __name__ == '__main__':
    tot1, rad1, tot2, rad2, x = main()
    
    print("\n=== Simulation Summary ===")
    print(f"System: 4 nm Au nanosphere, 1 nm above substrate")
    print(f"Substrate: Glass (n = {np.sqrt(2.25):.2f})")
    print(f"Dipole positions: x = 0 - 5 nm, z = 0.5 nm (between sphere and substrate)")
    print(f"Wavelength: 550 nm")
    print(f"Calculations: Quasistatic vs Full retarded (with substrate)")
    
    # Detailed analysis
    analyze_substrate_effects(x, tot1, rad1, tot2, rad2)