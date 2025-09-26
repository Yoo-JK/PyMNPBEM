"""
DEMODIPRET4 - Photonic LDOS for nanotriangle.

For a silver nanotriangle with 80 nm base length and 10 nm height,
this program computes the photonic LDOS for selected positions
(corner, center, edge, 5 nm above nanoparticle) and for different
transition dipole energies using the full Maxwell equations.

Runtime: ~3.5 min.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import polygon, polygon3, plate, vribbon, comparticle
from pymnpbem.mesh2d import edgeprofile
from pymnpbem.misc import compoint, units
from pymnpbem.bem import dipole, bemsolver


def refinement_function(pos):
    """Refinement function for mesh"""
    return 2 + 0.8 * np.abs(pos[:, 1]) ** 2


def main():
    # Initialization
    # Options for BEM simulation
    op = bemoptions(sim='ret', interp='curv')
    
    # Table of dielectric functions
    epstab = [epsconst(1), epstable('silver.dat')]
    
    # Dimensions of particle
    length = [80, 80 * 2 / np.sqrt(3)]
    
    # Polygon (triangle)
    poly = polygon(3, size=length).round()
    
    # Edge profile
    edge = edgeprofile(10, 11)
    
    # Upper plate for polygon, we only refine the upper plate
    p1, poly = plate(polygon3(poly, edge.zmax), edge=edge, refun=refinement_function)
    
    # Ribbon around polygon
    p2 = vribbon(poly)
    
    # Lower plate
    p3 = plate(poly.set_z(edge.zmin), dir=-1)
    
    # Set up COMPARTICLE objects
    p_geometry = np.vstack([p1, p2, p3])
    p = comparticle(epstab, [p_geometry], [2, 1], 1, op)
    
    # Dipole oscillator
    # Transition energies (eV)
    ene = np.linspace(1.5, 4, 40)
    
    # Transform to wavelengths
    unit_converter = units()
    enei = unit_converter.eV2nm / ene
    
    # Positions: corner, center, edge
    x = np.array([-45, 0, 25])
    
    # Compoint - positions 5 nm above the nanoparticle
    positions = np.column_stack([x, np.zeros_like(x), np.full_like(x, 10)])
    pt = compoint(p, positions)
    
    # Dipole excitation
    dip = dipole(pt, op)
    
    # Initialize total and radiative scattering rate
    tot = np.zeros((len(enei), len(x), 3))
    rad = np.zeros((len(enei), len(x), 3))
    
    # BEM simulation
    # Set up BEM solver
    bem = bemsolver(p, op)
    
    print("Running BEM simulation for silver nanotriangle...")
    
    # Loop over wavelengths
    for ien in tqdm(range(len(enei)), desc="BEM solver", ncols=80):
        # Surface charges
        sig = bem.solve(dip(p, enei[ien]))
        
        # Total and radiative decay rate
        tot[ien, :, :], rad[ien, :, :] = dip.decayrate(sig)
    
    print("BEM simulation completed!")
    
    # Final plot
    # Plot photonic LDOS at different positions
    plt.figure(figsize=(12, 8))
    
    # Average over all polarizations
    ldos = np.sum(tot, axis=2) / 3
    
    # Position labels
    pos_labels = ['Corner', 'Center', 'Edge']
    colors = plt.cm.tab10(np.arange(len(x)))
    
    for i in range(len(x)):
        plt.plot(ene, ldos[:, i], '-', color=colors[i], 
                label=pos_labels[i], linewidth=2, marker='o', markersize=4)
    
    plt.xlabel('Transition dipole energy (eV)')
    plt.ylabel('Photonic LDOS enhancement')
    plt.legend()
    plt.title('Photonic LDOS for silver nanotriangle (80 nm base, 10 nm height)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Additional analysis plot - separate subplots for each position
    plt.figure(figsize=(15, 5))
    
    for i in range(len(x)):
        plt.subplot(1, 3, i+1)
        plt.plot(ene, ldos[:, i], 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Energy (eV)')
        plt.ylabel('LDOS enhancement')
        plt.title(f'{pos_labels[i]} (x = {x[i]} nm)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Polarization-resolved plot
    plt.figure(figsize=(12, 8))
    polarizations = ['x', 'y', 'z']
    
    for i in range(len(x)):
        plt.subplot(2, 2, i+1 if i < 3 else 4)
        for j in range(3):
            plt.plot(ene, tot[:, i, j], '-', label=f'{polarizations[j]}-pol', 
                    linewidth=2, marker='o', markersize=3)
        plt.xlabel('Energy (eV)')
        plt.ylabel('LDOS enhancement')
        plt.title(f'{pos_labels[i]} - Polarization resolved')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return tot, rad, ene, enei, x


def analyze_triangle_resonances(tot, ene, x):
    """Analyze resonance characteristics for triangle geometry"""
    
    ldos = np.sum(tot, axis=2) / 3
    pos_labels = ['Corner', 'Center', 'Edge']
    
    print("\n=== Triangle Resonance Analysis ===")
    
    for i in range(len(x)):
        # Find all local maxima
        peaks = []
        for j in range(1, len(ene)-1):
            if ldos[j, i] > ldos[j-1, i] and ldos[j, i] > ldos[j+1, i]:
                if ldos[j, i] > 1.5:  # Only significant peaks
                    peaks.append((ene[j], ldos[j, i]))
        
        print(f"{pos_labels[i]} position (x = {x[i]} nm):")
        if peaks:
            for k, (peak_energy, peak_value) in enumerate(peaks):
                print(f"  Peak {k+1}: {peak_energy:.2f} eV, LDOS = {peak_value:.2f}")
        else:
            max_idx = np.argmax(ldos[:, i])
            print(f"  Maximum: {ene[max_idx]:.2f} eV, LDOS = {ldos[max_idx, i]:.2f}")
        print()


if __name__ == '__main__':
    tot, rad, ene, enei, x = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Energy range: {ene[0]:.1f} - {ene[-1]:.1f} eV")
    print(f"Wavelength range: {enei[-1]:.0f} - {enei[0]:.0f} nm")
    print(f"Triangle: 80 nm base length, 10 nm height")
    print(f"Positions: Corner (-45 nm), Center (0 nm), Edge (25 nm)")
    
    # Analyze resonance characteristics
    analyze_triangle_resonances(tot, ene, x)