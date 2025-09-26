"""
DEMODIPSTAT5 - Lifetime reduction for dipole between sphere and layer.

For a metallic nanosphere with a diameter of 4 nm located 1 nm above a
substrate and a dipole located between sphere and layer, this program
computes the total dipole scattering rates within the quasistatic
approximation.

Runtime: ~7.5 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import trispheresegment, comparticle, shift, flip
from pymnpbem.misc import compoint, layerstructure
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
    
    # Options for BEM simulation (quasistatic with layer)
    op = bemoptions(sim='stat', interp='curv', layer=layer)
    
    # Nanosphere with finer discretization at the bottom
    phi_grid = 2 * np.pi * np.linspace(0, 1, 31)
    theta_grid = np.pi * np.linspace(0, 1, 31) ** 2
    p_sphere = trispheresegment(phi_grid, theta_grid, 4)  # 4 nm diameter
    
    # Flip sphere (finer discretization at bottom)
    p_sphere = flip(p_sphere, axis=2)  # flip along z-axis
    
    # Place nanosphere 1 nm above substrate
    min_z = np.min(p_sphere.pos[:, 2])
    p_sphere = shift(p_sphere, [0, 0, -min_z + 1 + ztab])
    
    # Set up COMPARTICLE object
    p = comparticle(epstab, [p_sphere], [2, 1], 1, op)  # gold sphere in air
    
    # Dipole oscillator
    enei = 550
    
    # Positions of dipole (between sphere and substrate)
    x = np.linspace(0, 5, 51).reshape(-1, 1)
    
    # Compoint - dipoles at z = 0.5 nm (between sphere and substrate)
    positions = np.column_stack([x.flatten(), np.zeros_like(x.flatten()), 
                                np.full_like(x.flatten(), 0.5)])
    pt = compoint(p, positions, op)
    
    print("Running quasistatic BEM simulation with substrate...")
    
    # Dipole excitation
    dip = dipole(pt, np.array([[1, 0, 0], [0, 0, 1]]), op)
    
    # BEM simulation
    # Set up BEM solver
    bem = bemsolver(p, op)
    
    # Surface charge
    sig = bem.solve(dip(p, enei))
    
    # Total and radiative decay rate
    tot, rad = dip.decayrate(sig)
    
    print("Quasistatic simulation with substrate completed!")
    
    # Final plot
    plt.figure(figsize=(12, 8))
    
    # Plot total decay rates
    plt.plot(x.flatten(), tot[:, 0], '-', label='x-dip', linewidth=2, marker='o', markersize=4)
    plt.plot(x.flatten(), tot[:, 1], '-', label='z-dip', linewidth=2, marker='s', markersize=4)
    
    plt.xlabel('Position (nm)')
    plt.ylabel('Total decay rate')
    plt.legend()
    plt.title('Quasistatic decay rates: 4nm Au sphere 1nm above substrate')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    analyze_substrate_coupling(x, tot, rad)
    
    return tot, rad, x


def analyze_substrate_coupling(x, tot, rad):
    """Analyze substrate coupling effects in quasistatic regime"""
    
    print("\n=== Quasistatic Substrate Coupling Analysis ===")
    
    # Enhancement factors
    for pol, label in enumerate(['x', 'z']):
        print(f"\n{label.upper()}-polarization:")
        
        # Maximum enhancement
        max_idx = np.argmax(tot[:, pol])
        max_position = x[max_idx, 0]
        max_enhancement = tot[max_idx, pol]
        
        print(f"  Maximum enhancement: {max_enhancement:.2f} at x = {max_position:.2f} nm")
        
        # Enhancement at contact (x = 0)
        contact_enhancement = tot[0, pol]
        print(f"  At contact (x = 0): {contact_enhancement:.2f}")
        
        # Enhancement decay
        if len(x) > 10:
            mid_idx = len(x) // 2
            far_idx = -1
            
            mid_enhancement = tot[mid_idx, pol]
            far_enhancement = tot[far_idx, pol]
            
            print(f"  At x = {x[mid_idx, 0]:.1f} nm: {mid_enhancement:.2f}")
            print(f"  At x = {x[far_idx, 0]:.1f} nm: {far_enhancement:.2f}")
            
            # Decay ratio
            decay_ratio = contact_enhancement / far_enhancement
            print(f"  Decay ratio (contact/far): {decay_ratio:.1f}")
    
    # Substrate vs free space comparison (conceptual)
    print(f"\nSubstrate effects:")
    print(f"In quasistatic approximation with substrate:")
    print(f"- Enhanced local field due to substrate reflection")
    print(f"- Modified boundary conditions at substrate interface") 
    print(f"- No radiative effects (quasistatic regime)")
    
    # Distance analysis
    print(f"\nGeometry:")
    print(f"- Sphere diameter: 4 nm")
    print(f"- Sphere height above substrate: 1 nm")
    print(f"- Dipole height: 0.5 nm (between sphere and substrate)")
    print(f"- Gap size at contact: 0.5 nm")


def plot_detailed_quasistatic_analysis(x, tot, rad):
    """Detailed analysis of quasistatic results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total decay rates
    axes[0, 0].plot(x.flatten(), tot[:, 0], 'r-o', label='x-polarization', 
                   linewidth=2, markersize=4)
    axes[0, 0].plot(x.flatten(), tot[:, 1], 'b-s', label='z-polarization', 
                   linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Position (nm)')
    axes[0, 0].set_ylabel('Total decay rate')
    axes[0, 0].set_title('Total decay rates (quasistatic)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Radiative decay rates
    axes[0, 1].plot(x.flatten(), rad[:, 0], 'r-o', label='x-polarization', 
                   linewidth=2, markersize=4)
    axes[0, 1].plot(x.flatten(), rad[:, 1], 'b-s', label='z-polarization', 
                   linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Position (nm)')
    axes[0, 1].set_ylabel('Radiative decay rate')
    axes[0, 1].set_title('Radiative decay rates')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Non-radiative decay rates
    nonrad = tot - rad
    axes[1, 0].plot(x.flatten(), nonrad[:, 0], 'r-o', label='x-polarization', 
                   linewidth=2, markersize=4)
    axes[1, 0].plot(x.flatten(), nonrad[:, 1], 'b-s', label='z-polarization', 
                   linewidth=2, markersize=4)
    axes[1, 0].set_xlabel('Position (nm)')
    axes[1, 0].set_ylabel('Non-radiative decay rate')
    axes[1, 0].set_title('Non-radiative decay rates')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Quantum efficiency
    qe = rad / (tot + 1e-10)
    axes[1, 1].plot(x.flatten(), qe[:, 0], 'r-o', label='x-polarization', 
                   linewidth=2, markersize=4)
    axes[1, 1].plot(x.flatten(), qe[:, 1], 'b-s', label='z-polarization', 
                   linewidth=2, markersize=4)
    axes[1, 1].set_xlabel('Position (nm)')
    axes[1, 1].set_ylabel('Quantum efficiency')
    axes[1, 1].set_title('Radiative quantum efficiency')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_polarizations(x, tot, rad):
    """Compare x and z polarization responses"""
    
    print("\n=== Polarization Comparison (Quasistatic) ===")
    
    # Find crossover points where one polarization dominates
    x_dominant = tot[:, 0] > tot[:, 1]
    z_dominant = tot[:, 1] > tot[:, 0]
    
    if np.any(x_dominant) and np.any(z_dominant):
        # Find transition point
        transitions = np.where(np.diff(x_dominant.astype(int)) != 0)[0]
        if len(transitions) > 0:
            transition_x = x[transitions[0], 0]
            print(f"Polarization dominance transition near x = {transition_x:.2f} nm")
    
    # Maximum enhancement ratios
    ratio_x_to_z = np.max(tot[:, 0] / (tot[:, 1] + 1e-10))
    ratio_z_to_x = np.max(tot[:, 1] / (tot[:, 0] + 1e-10))
    
    print(f"Maximum enhancement ratios:")
    print(f"  x/z polarization: {ratio_x_to_z:.2f}")
    print(f"  z/x polarization: {ratio_z_to_x:.2f}")
    
    # Average quantum efficiencies
    qe = rad / (tot + 1e-10)
    avg_qe_x = np.mean(qe[:, 0])
    avg_qe_z = np.mean(qe[:, 1])
    
    print(f"Average quantum efficiencies:")
    print(f"  x-polarization: {avg_qe_x:.3f}")
    print(f"  z-polarization: {avg_qe_z:.3f}")


if __name__ == '__main__':
    tot, rad, x = main()
    
    print("\n=== Simulation Summary ===")
    print(f"System: 4 nm Au nanosphere, 1 nm above glass substrate")
    print(f"Substrate: Glass (n = {np.sqrt(2.25):.2f})")
    print(f"Dipole positions: x = 0 - 5 nm, z = 0.5 nm")
    print(f"Wavelength: 550 nm")
    print(f"Method: Quasistatic approximation with substrate")
    print(f"Key advantage: Fast calculation, good for small particles")
    
    # Detailed analysis
    plot_detailed_quasistatic_analysis(x, tot, rad)
    compare_polarizations(x, tot, rad)