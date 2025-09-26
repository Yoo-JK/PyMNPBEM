"""
DEMODIPRET3 - Photonic LDOS for a silver nanodisk.

For a silver nanodisk with a diameter of 30 nm and a height of 5 nm,
we compute the photonic LDOS for selected positions (0, R/2, R, 3R/2,
2.5 nm above nanoparticle) and for different transition dipole
energies using the full Maxwell equations, and compare the results
with those obtained within the quasistatic approximation.

Runtime: ~2 min.
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
    """
    Refinement function for mesh
    refun = @( pos, ~ ) 0.8 + abs( pos( :, 2 ) ) .^ 2 + 5 * ( pos( :, 1 ) < 0 );
    """
    return 0.8 + np.abs(pos[:, 1]) ** 2 + 5 * (pos[:, 0] < 0)


def main():
    # Initialization
    # Options for BEM simulation
    op1 = bemoptions(sim='stat', interp='curv')  # Quasistatic
    op2 = bemoptions(sim='ret', interp='curv')   # Retarded
    
    # Table of dielectric functions
    epstab = [epsconst(1), epstable('silver.dat')]
    
    # Polygon for disk
    poly = polygon(25, size=[30, 30])
    
    # Edge profile for triangle
    edge = edgeprofile(5, 11)
    
    # Upper plate for polygon, we only refine the upper plate
    p1, poly = plate(polygon3(poly, edge.zmax), edge=edge, refun=refinement_function)
    
    # Ribbon around polygon
    p2 = vribbon(poly)
    
    # Lower plate
    p3 = plate(poly.set_z(edge.zmin), dir=-1)
    
    # Set up COMPARTICLE objects
    p_geometry = np.vstack([p1, p2, p3])  # vertcat equivalent
    p = comparticle(epstab, [p_geometry], [2, 1], 1, op1)
    
    # Dipole oscillator
    # Transition energies (eV)
    ene = np.linspace(2.5, 4, 60)
    
    # Transform to wavelengths
    unit_converter = units()
    enei = unit_converter.eV2nm / ene  # Convert eV to nm
    
    # Positions
    x = np.array([0, 6, 12, 18])  # 0, R/2, R, 3R/2 where R ≈ 15 nm (disk radius)
    
    # Compoint - positions 2.5 nm above the nanoparticle
    positions = np.column_stack([x, np.zeros_like(x), np.full_like(x, 5)])
    pt = compoint(p, positions)
    
    # Dipole excitations
    dip1 = dipole(pt, op1)  # Quasistatic
    dip2 = dipole(pt, op2)  # Retarded
    
    # Initialize total and radiative scattering rate
    # Shape: (energy, position, polarization)
    tot1 = np.zeros((len(enei), len(x), 3))
    rad1 = np.zeros((len(enei), len(x), 3))
    tot2 = np.zeros((len(enei), len(x), 3))
    rad2 = np.zeros((len(enei), len(x), 3))
    
    # BEM simulation
    # Set up BEM solvers
    bem1 = bemsolver(p, op1)  # Quasistatic solver
    bem2 = bemsolver(p, op2)  # Retarded solver
    
    print("Running BEM simulation (both quasistatic and retarded)...")
    
    # Loop over wavelengths with progress bar
    for ien in tqdm(range(len(enei)), desc="BEM solver", ncols=80):
        # Surface charges
        sig1 = bem1.solve(dip1(p, enei[ien]))  # Quasistatic
        sig2 = bem2.solve(dip2(p, enei[ien]))  # Retarded
        
        # Total and radiative decay rate
        tot1[ien, :, :], rad1[ien, :, :] = dip1.decayrate(sig1)
        tot2[ien, :, :], rad2[ien, :, :] = dip2.decayrate(sig2)
    
    print("BEM simulation completed!")
    
    # Final plot
    # Plot photonic LDOS at different positions
    plt.figure(figsize=(12, 8))
    
    # Average over all polarizations (sum over 3rd dimension, divide by 3)
    ldos1 = np.sum(tot1, axis=2) / 3  # Quasistatic LDOS
    ldos2 = np.sum(tot2, axis=2) / 3  # Retarded LDOS
    
    # Position labels
    pos_labels = ['0', 'R/2', 'R', '3R/2']
    colors = plt.cm.tab10(np.arange(len(x)))
    
    # Plot quasistatic results (solid lines)
    for i in range(len(x)):
        plt.plot(ene, ldos1[:, i], '-', color=colors[i], 
                label=f'{pos_labels[i]} (stat)', linewidth=2)
    
    # Plot retarded results (dashed lines)
    for i in range(len(x)):
        plt.plot(ene, ldos2[:, i], '--', color=colors[i], 
                label=f'{pos_labels[i]} (ret)', linewidth=2)
    
    plt.xlabel('Transition dipole energy (eV)')
    plt.ylabel('Photonic LDOS enhancement')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Photonic LDOS for silver nanodisk (30 nm diameter, 5 nm height)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Additional analysis plot - comparing quasistatic vs retarded
    plt.figure(figsize=(12, 6))
    
    # Plot comparison for each position
    for i in range(len(x)):
        plt.subplot(2, 2, i+1)
        plt.plot(ene, ldos1[:, i], '-', label='Quasistatic', linewidth=2)
        plt.plot(ene, ldos2[:, i], '--', label='Retarded', linewidth=2)
        plt.xlabel('Energy (eV)')
        plt.ylabel('LDOS enhancement')
        plt.title(f'Position: {pos_labels[i]} (x = {x[i]} nm)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return tot1, rad1, tot2, rad2, ene, enei, x


def analyze_resonances(tot1, tot2, ene, x):
    """Analyze resonance peaks in the LDOS spectra"""
    
    # Average LDOS over polarizations
    ldos1 = np.sum(tot1, axis=2) / 3
    ldos2 = np.sum(tot2, axis=2) / 3
    
    print("\n=== Resonance Analysis ===")
    pos_labels = ['0', 'R/2', 'R', '3R/2']
    
    for i in range(len(x)):
        # Find peaks in quasistatic and retarded spectra
        peak_idx1 = np.argmax(ldos1[:, i])
        peak_idx2 = np.argmax(ldos2[:, i])
        
        peak_energy1 = ene[peak_idx1]
        peak_energy2 = ene[peak_idx2]
        peak_value1 = ldos1[peak_idx1, i]
        peak_value2 = ldos2[peak_idx2, i]
        
        print(f"Position {pos_labels[i]} (x = {x[i]} nm):")
        print(f"  Quasistatic peak: {peak_energy1:.2f} eV, LDOS = {peak_value1:.2f}")
        print(f"  Retarded peak: {peak_energy2:.2f} eV, LDOS = {peak_value2:.2f}")
        print(f"  Peak shift: {peak_energy2 - peak_energy1:.3f} eV")
        print(f"  Enhancement ratio (ret/stat): {peak_value2/peak_value1:.2f}")
        print()


def plot_surface_charge(p, sig, title="Surface charge distribution"):
    """Plot surface charge distribution on the nanoparticle"""
    # This would be implemented based on the specific visualization capabilities
    # of your PyMNPBEM implementation
    print(f"Surface charge plotting: {title}")
    # Placeholder for actual implementation
    # plot(p, sig.sig)  # MATLAB equivalent


if __name__ == '__main__':
    tot1, rad1, tot2, rad2, ene, enei, x = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Energy range: {ene[0]:.1f} - {ene[-1]:.1f} eV")
    print(f"Wavelength range: {enei[-1]:.0f} - {enei[0]:.0f} nm")
    print(f"Number of energy points: {len(ene)}")
    print(f"Nanodisk: 30 nm diameter, 5 nm height")
    print(f"Dipole positions: {x} nm (2.5 nm above surface)")
    
    # Analyze resonance peaks
    analyze_resonances(tot1, tot2, ene, x)
    
    # Optional: plot surface charge distribution for the last calculated wavelength
    # This would require the last calculated sig1 and sig2 to be available
    print("\nNote: Surface charge plotting can be enabled by uncommenting the relevant lines in the original code.")