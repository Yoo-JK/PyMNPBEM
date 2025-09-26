"""
DEMOEELSRET3 - Induced electric field for EELS of nanodisk.

For a silver nanodisk with 60 nm diameter and 10 nm height, this
program computes the induced electric field for an electron beam
excitation for two selected impact parameters (inside and outside of
disk) along the electron beam trajectory, using the full Maxwell
equations.

Runtime: ~8 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import polygon, tripolygon, comparticle
from pymnpbem.mesh2d import edgeprofile
from pymnpbem.misc import units, eelsbase, compoint, greenfunction
from pymnpbem.bem import bemsolver, electronbeam


def main():
    # Initialization
    # Options for BEM simulation
    op = bemoptions(sim='ret', interp='curv')
    
    # Table of dielectric function
    epstab = [epsconst(1), epstable('silver.dat')]
    
    # Diameter of disk
    diameter = 60
    
    # Polygon for disk
    poly = polygon(25, size=[diameter, diameter])
    
    # Edge profile for disk
    edge = edgeprofile(10, 11)
    
    # Extrude polygon to nanoparticle
    p = comparticle(epstab, [tripolygon(poly, edge)], [2, 1], 1, op)
    
    # Width of electron beam and electron velocity
    width = 0.5
    vel = eelsbase.ene2vel(200e3)  # 200 keV electron energy
    
    # Impact parameters (inside and outside of disk)
    imp = np.array([0.9, 1.2]) * diameter / 2  # 27 nm, 36 nm
    
    # Loss energy in eV
    ene = 2.6
    
    # Convert energy to nm
    unit_converter = units()
    enei = unit_converter.eV2nm / ene
    
    print("Running BEM simulation for electric field along electron trajectories...")
    
    # BEM simulation
    # Set up BEM solver
    bem = bemsolver(p, enei, op)
    
    # EELS excitation
    exc1 = electronbeam(p, [imp[0], 0], width, vel, op)  # Inside disk
    exc2 = electronbeam(p, [imp[1], 0], width, vel, op)  # Outside disk
    
    # z-values where field is computed
    z = np.linspace(-80, 80, 1001)
    
    # Convert to points
    pt1 = compoint(p, np.column_stack([np.full_like(z, imp[0]), 
                                      np.zeros_like(z), z]), mindist=0.1)
    pt2 = compoint(p, np.column_stack([np.full_like(z, imp[1]), 
                                      np.zeros_like(z), z]), mindist=0.1)
    
    # Green function objects
    g1 = greenfunction(pt1, p, op)
    g2 = greenfunction(pt2, p, op)
    
    # Compute surface charges
    sig1 = bem.solve(exc1(enei))
    sig2 = bem.solve(exc2(enei))
    
    # Compute fields
    field1 = g1.field(sig1)
    field2 = g2.field(sig2)
    
    # Electric field
    e1 = pt1(field1.e)
    e2 = pt2(field2.e)
    
    print("Electric field calculation completed!")
    
    # Final plot
    plt.figure(figsize=(12, 8))
    
    # Plot z-component of electric field
    plt.plot(z, np.real(e1[:, 2]), label='Inside disk', linewidth=2)
    plt.plot(z, np.real(e2[:, 2]), label='Outside disk', linewidth=2)
    
    # Mark disk boundaries
    plt.plot([-5, -5], [-4, 4], 'k--', alpha=0.7, linewidth=1)
    plt.plot([5, 5], [-4, 4], 'k--', alpha=0.7, linewidth=1)
    
    plt.ylim([-4, 4])
    plt.legend()
    plt.xlabel('z (nm)')
    plt.ylabel('Electric field (Re[E_z])')
    plt.title(f'Electric field along electron trajectories (λ = {ene:.1f} eV)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    analyze_field_distribution(z, e1, e2, imp, diameter)
    
    return e1, e2, z, imp


def analyze_field_distribution(z, e1, e2, imp, diameter):
    """Analyze electric field distribution characteristics"""
    
    print("\n=== Electric Field Analysis ===")
    
    # Impact parameters
    print(f"Impact parameters:")
    print(f"  Inside disk: {imp[0]:.1f} nm")
    print(f"  Outside disk: {imp[1]:.1f} nm")
    print(f"  Disk radius: {diameter/2:.1f} nm")
    
    # Field maxima
    max_field_inside = np.max(np.abs(np.real(e1[:, 2])))
    max_field_outside = np.max(np.abs(np.real(e2[:, 2])))
    
    max_pos_inside = z[np.argmax(np.abs(np.real(e1[:, 2])))]
    max_pos_outside = z[np.argmax(np.abs(np.real(e2[:, 2])))]
    
    print(f"\nMaximum field amplitudes:")
    print(f"  Inside trajectory: {max_field_inside:.2f} at z = {max_pos_inside:.1f} nm")
    print(f"  Outside trajectory: {max_field_outside:.2f} at z = {max_pos_outside:.1f} nm")
    
    # Field at disk boundaries
    disk_boundary_indices = [np.argmin(np.abs(z - (-5))), np.argmin(np.abs(z - 5))]
    
    print(f"\nField at disk boundaries:")
    for i, boundary_z in enumerate([-5, 5]):
        idx = disk_boundary_indices[i]
        field_inside = np.real(e1[idx, 2])
        field_outside = np.real(e2[idx, 2])
        print(f"  At z = {boundary_z} nm:")
        print(f"    Inside trajectory: {field_inside:.2f}")
        print(f"    Outside trajectory: {field_outside:.2f}")
    
    # Field enhancement ratio
    enhancement_ratio = max_field_inside / (max_field_outside + 1e-10)
    print(f"\nField enhancement ratio (inside/outside): {enhancement_ratio:.2f}")


def plot_detailed_field_analysis(z, e1, e2, imp, diameter):
    """Detailed field analysis plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Real and imaginary parts of E_z
    axes[0, 0].plot(z, np.real(e1[:, 2]), 'r-', label='Inside (real)', linewidth=2)
    axes[0, 0].plot(z, np.imag(e1[:, 2]), 'r--', label='Inside (imag)', linewidth=2)
    axes[0, 0].plot(z, np.real(e2[:, 2]), 'b-', label='Outside (real)', linewidth=2)
    axes[0, 0].plot(z, np.imag(e2[:, 2]), 'b--', label='Outside (imag)', linewidth=2)
    axes[0, 0].axvline(x=-5, color='k', linestyle=':', alpha=0.5)
    axes[0, 0].axvline(x=5, color='k', linestyle=':', alpha=0.5)
    axes[0, 0].set_xlabel('z (nm)')
    axes[0, 0].set_ylabel('E_z')
    axes[0, 0].set_title('z-component of electric field')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Field magnitude
    mag1 = np.sqrt(np.sum(np.abs(e1)**2, axis=1))
    mag2 = np.sqrt(np.sum(np.abs(e2)**2, axis=1))
    
    axes[0, 1].plot(z, mag1, 'r-', label='Inside disk', linewidth=2)
    axes[0, 1].plot(z, mag2, 'b-', label='Outside disk', linewidth=2)
    axes[0, 1].axvline(x=-5, color='k', linestyle=':', alpha=0.5)
    axes[0, 1].axvline(x=5, color='k', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('z (nm)')
    axes[0, 1].set_ylabel('|E|')
    axes[0, 1].set_title('Electric field magnitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # x and y components
    axes[1, 0].plot(z, np.real(e1[:, 0]), 'r-', label='E_x (inside)', linewidth=2)
    axes[1, 0].plot(z, np.real(e1[:, 1]), 'r--', label='E_y (inside)', linewidth=2)
    axes[1, 0].plot(z, np.real(e2[:, 0]), 'b-', label='E_x (outside)', linewidth=2)
    axes[1, 0].plot(z, np.real(e2[:, 1]), 'b--', label='E_y (outside)', linewidth=2)
    axes[1, 0].axvline(x=-5, color='k', linestyle=':', alpha=0.5)
    axes[1, 0].axvline(x=5, color='k', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('z (nm)')
    axes[1, 0].set_ylabel('E_x, E_y')
    axes[1, 0].set_title('Transverse field components')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Field ratio
    field_ratio = np.abs(e1[:, 2]) / (np.abs(e2[:, 2]) + 1e-10)
    axes[1, 1].plot(z, field_ratio, 'g-', linewidth=2)
    axes[1, 1].axvline(x=-5, color='k', linestyle=':', alpha=0.5)
    axes[1, 1].axvline(x=5, color='k', linestyle=':', alpha=0.5)
    axes[1, 1].axhline(y=1, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('z (nm)')
    axes[1, 1].set_ylabel('|E_z(inside)|/|E_z(outside)|')
    axes[1, 1].set_title('Field enhancement ratio')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def explain_eels_field_physics():
    """Explain the physics of EELS field enhancement"""
    
    print("\n=== EELS Field Physics ===")
    print("Field enhancement mechanisms:")
    print("- Inside trajectory: Strong coupling to all multipolar modes")
    print("- Outside trajectory: Mainly dipolar surface plasmon coupling")
    print("- Field maximum typically near particle boundaries")
    print("- Plasmon resonances enhance local fields")
    print("- Different impact parameters probe different modes")
    print("- Field asymmetry reveals particle-electron interaction")


if __name__ == '__main__':
    e1, e2, z, imp = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Silver nanodisk: 60 nm diameter, 10 nm height")
    print(f"Electron beam: 200 keV energy")
    print(f"Loss energy: 2.6 eV")
    print(f"Impact parameters: {imp[0]:.1f} nm (inside), {imp[1]:.1f} nm (outside)")
    print(f"Field calculation: Along z-axis from -80 to +80 nm")
    
    # Detailed analysis
    plot_detailed_field_analysis(z, e1, e2, imp, 60)
    explain_eels_field_physics()