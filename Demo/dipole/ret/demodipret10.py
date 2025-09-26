"""
DEMODIPRET10 - Photonic LDOS for nanodisk above layer.

For a silver nanodisk with a diameter of 30 nm and a height of 5 nm,
and a dipole oscillator with dipole orientation along x and z and
located 5 nm away from the disk and 0.5 nm above the substrate, this
program computes the lifetime reduction (photonic LDOS) as a function
of transition dipole energies, using the full Maxwell equations.

Runtime: ~7 min.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pymnpbem.base import bemoptions
from pymnpbem.material import epsconst, epstable
from pymnpbem.particles import polygon, tripolygon, comparticle
from pymnpbem.mesh2d import edgeprofile
from pymnpbem.misc import compoint, layerstructure, tabspace, compgreentablayer
from pymnpbem.bem import dipole, bemsolver


def refinement_function(pos, d):
    """Refinement function for finer discretization at (R,0,0)"""
    return 0.5 + np.abs(15 - pos[:, 0])


def main():
    # Initialization
    # Table of dielectric functions
    epstab = [epsconst(1), epstable('silver.dat'), epsconst(4)]  # air, silver, substrate
    
    # Location of interface of substrate
    ztab = 0
    
    # Default options for layer structure
    opt = layerstructure.options()
    
    # Set up layer structure
    layer = layerstructure(epstab, [1, 3], ztab, opt)  # air above, substrate below
    
    # Options for BEM simulation
    op = bemoptions(sim='ret', interp='curv', layer=layer)
    
    # Polygon for disk
    poly = polygon(25, size=[30, 30])
    
    # Edge profile for nanodisk
    # MODE '01' produces rounded edge on top and sharp edge on bottom
    # MIN controls the lower z-value of the nanoparticle
    edge = edgeprofile(5, 11, mode='01', min=1e-3)
    
    # Extrude polygon to nanoparticle
    p = tripolygon(poly, edge, refun=refinement_function)
    
    # Set up COMPARTICLE object
    p = comparticle(epstab, [p], [2, 1], 1, op)  # silver disk in air
    
    # Dipole oscillator
    enei = np.linspace(300, 700, 40)
    
    # Position of dipole (5 nm away from disk edge, 0.5 nm above substrate)
    x = np.max(p.pos[:, 0]) + 2  # 2 nm from edge (approximately 5 nm from center)
    
    # Compoint
    pt = compoint(p, [x, 0, 0.5], op)  # 0.5 nm above substrate
    
    print("Computing Green's function table for nanodisk above substrate...")
    print("This may take several minutes for the first run...")
    
    # Tabulated Green functions
    # For retarded simulation, set up table for reflected Green function
    
    # Automatic grid for tabulation (small NZ for speed)
    tab = tabspace(layer, p, pt, nz=5)
    
    # Green function table
    greentab = compgreentablayer(layer, tab)
    
    # Precompute Green function table
    # For more accurate simulation, increase number of wavelengths
    wavelength_table = np.linspace(300, 800, 5)
    greentab = greentab.set(wavelength_table, op)
    
    # Add Green table to options
    op.greentab = greentab
    
    print("Green's function table completed!")
    
    # Dipole excitation
    dip = dipole(pt, np.array([[1, 0, 0], [0, 0, 1]]), op)  # x and z polarizations
    
    # BEM simulation
    # Set up BEM solver
    bem = bemsolver(p, op)
    
    # Initialize total and radiative scattering rate
    tot = np.zeros((len(enei), 2))
    rad = np.zeros((len(enei), 2))
    
    print("Running BEM simulation over wavelength range...")
    
    # Loop over wavelengths
    for ien in tqdm(range(len(enei)), desc="BEM solver", ncols=80):
        # Surface charge
        sig = bem.solve(dip(p, enei[ien]))
        
        # Total and radiative decay rate
        tot[ien, :], rad[ien, :] = dip.decayrate(sig)
    
    print("BEM simulation completed!")
    
    # Final plot
    plt.figure(figsize=(12, 8))
    
    # Plot total decay rates
    plt.plot(enei, tot[:, 0], '.-', label='x-dip', linewidth=2, markersize=6)
    plt.plot(enei, tot[:, 1], '.-', label='z-dip', linewidth=2, markersize=6)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Total decay rate')
    plt.legend()
    plt.title('Photonic LDOS for dipole near silver nanodisk above substrate')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Additional analysis plots
    plot_detailed_analysis(enei, tot, rad)
    
    return tot, rad, enei, x


def plot_detailed_analysis(enei, tot, rad):
    """Detailed analysis of LDOS and quantum efficiency"""
    
    # Calculate non-radiative decay and quantum efficiency
    nonrad = tot - rad
    qe = rad / tot
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total vs radiative decay rates
    axes[0, 0].plot(enei, tot[:, 0], 'r-', label='Total (x)', linewidth=2)
    axes[0, 0].plot(enei, rad[:, 0], 'r--', label='Radiative (x)', linewidth=2)
    axes[0, 0].plot(enei, nonrad[:, 0], 'r:', label='Non-radiative (x)', linewidth=2)
    axes[0, 0].set_xlabel('Wavelength (nm)')
    axes[0, 0].set_ylabel('Decay rate')
    axes[0, 0].set_title('x-polarized dipole')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(enei, tot[:, 1], 'b-', label='Total (z)', linewidth=2)
    axes[0, 1].plot(enei, rad[:, 1], 'b--', label='Radiative (z)', linewidth=2)
    axes[0, 1].plot(enei, nonrad[:, 1], 'b:', label='Non-radiative (z)', linewidth=2)
    axes[0, 1].set_xlabel('Wavelength (nm)')
    axes[0, 1].set_ylabel('Decay rate')
    axes[0, 1].set_title('z-polarized dipole')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Quantum efficiency
    axes[1, 0].plot(enei, qe[:, 0], 'r-', label='x-polarization', linewidth=2)
    axes[1, 0].plot(enei, qe[:, 1], 'b-', label='z-polarization', linewidth=2)
    axes[1, 0].set_xlabel('Wavelength (nm)')
    axes[1, 0].set_ylabel('Quantum efficiency')
    axes[1, 0].set_title('Radiative quantum efficiency')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Enhancement factors comparison
    axes[1, 1].plot(enei, tot[:, 0], 'r-', label='Total (x)', linewidth=2)
    axes[1, 1].plot(enei, tot[:, 1], 'b-', label='Total (z)', linewidth=2)
    axes[1, 1].axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Free space')
    axes[1, 1].set_xlabel('Wavelength (nm)')
    axes[1, 1].set_ylabel('LDOS enhancement')
    axes[1, 1].set_title('Total LDOS enhancement')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_resonance_peaks(enei, tot, rad):
    """Analyze resonance peaks and their characteristics"""
    
    print("\n=== Resonance Peak Analysis ===")
    
    for pol, label in enumerate(['x', 'z']):
        print(f"\n{label.upper()}-polarization:")
        
        # Find local maxima in total decay rate
        peaks = []
        for i in range(2, len(enei)-2):
            if (tot[i, pol] > tot[i-1, pol] and tot[i, pol] > tot[i+1, pol] and
                tot[i, pol] > tot[i-2, pol] and tot[i, pol] > tot[i+2, pol]):
                if tot[i, pol] > 2.0:  # Only significant peaks
                    peaks.append((enei[i], tot[i, pol], rad[i, pol]))
        
        if peaks:
            for j, (wavelength, tot_val, rad_val) in enumerate(peaks):
                qe = rad_val / tot_val
                print(f"  Peak {j+1}: λ = {wavelength:.0f} nm")
                print(f"    Total LDOS: {tot_val:.2f}")
                print(f"    Radiative LDOS: {rad_val:.2f}")
                print(f"    Quantum efficiency: {qe:.3f}")
        else:
            # If no clear peaks, show maximum
            max_idx = np.argmax(tot[:, pol])
            max_wavelength = enei[max_idx]
            max_tot = tot[max_idx, pol]
            max_rad = rad[max_idx, pol]
            max_qe = max_rad / max_tot
            
            print(f"  Maximum enhancement: λ = {max_wavelength:.0f} nm")
            print(f"    Total LDOS: {max_tot:.2f}")
            print(f"    Radiative LDOS: {max_rad:.2f}")
            print(f"    Quantum efficiency: {max_qe:.3f}")
        
        # Average quantum efficiency
        avg_qe = np.mean(rad[:, pol] / tot[:, pol])
        print(f"  Average quantum efficiency: {avg_qe:.3f}")


def compare_polarizations(enei, tot, rad):
    """Compare x and z polarization responses"""
    
    print("\n=== Polarization Comparison ===")
    
    # Find wavelength regions where one polarization dominates
    x_dominant = tot[:, 0] > tot[:, 1]
    z_dominant = tot[:, 1] > tot[:, 0]
    
    x_dominant_regions = np.where(x_dominant)[0]
    z_dominant_regions = np.where(z_dominant)[0]
    
    if len(x_dominant_regions) > 0:
        x_wavelengths = enei[x_dominant_regions]
        print(f"x-polarization dominant: λ = {x_wavelengths[0]:.0f}-{x_wavelengths[-1]:.0f} nm")
    
    if len(z_dominant_regions) > 0:
        z_wavelengths = enei[z_dominant_regions]
        print(f"z-polarization dominant: λ = {z_wavelengths[0]:.0f}-{z_wavelengths[-1]:.0f} nm")
    
    # Maximum enhancement ratio
    max_ratio_xz = np.max(tot[:, 0] / tot[:, 1])
    max_ratio_zx = np.max(tot[:, 1] / tot[:, 0])
    
    print(f"Maximum enhancement ratio (x/z): {max_ratio_xz:.2f}")
    print(f"Maximum enhancement ratio (z/x): {max_ratio_zx:.2f}")


if __name__ == '__main__':
    tot, rad, enei, dipole_x = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Silver nanodisk: 30 nm diameter, 5 nm height")
    print(f"Substrate: ε = 4 (n = 2.0)")
    print(f"Dipole position: ({dipole_x:.1f}, 0, 0.5) nm")
    print(f"Distance from disk edge: ~2 nm")
    print(f"Wavelength range: {enei[0]:.0f} - {enei[-1]:.0f} nm")
    print(f"Number of wavelength points: {len(enei)}")
    
    # Analyze results
    analyze_resonance_peaks(enei, tot, rad)
    compare_polarizations(enei, tot, rad)