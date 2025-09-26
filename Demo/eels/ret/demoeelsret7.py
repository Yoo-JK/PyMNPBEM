"""
DEMOEELSRET7 - EEL spectra for silver nanotriangle on membrane.

For a silver nanotriangle with 80 nm base length and 10 nm height,
which is located on a 15 nm thin membrane, this program computes the
energy loss probabilities for selected impact parameters using the
full Maxwell equations.

Runtime: ~9 min.
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
    # MODE '01' produces rounded edge on top and sharp edge on bottom
    edge = edgeprofile(10, 11, mode='01', min=0)
    
    # Mesh data structure
    hdata = {'hmax': 8}
    
    # Extrude polygon
    p, poly = tripolygon(poly, edge, hdata=hdata)
    
    print("Creating complex membrane geometry...")
    
    # Split into upper and lower part
    pup, plo = select(p, carfun=lambda x, y, z: z > 1e-3)
    
    # Polygon for plate (membrane)
    poly2 = polygon3(polygon(4, size=[150, 150]), 0)
    
    # Upper plate
    up = plate([poly2, poly.set_z(0)], dir=-1)
    
    # Lower plate, use FVGRID to get quadrilateral face elements
    x = 150 * np.linspace(-0.5, 0.5, 21)
    verts, faces = fvgrid(x, x)
    
    # Make particle
    lo = flipfaces(shift(particle(verts, faces), [0, 0, -15]))
    
    # Make complete particle system
    p = comparticle(epstab, [pup, plo, up, lo],
                   [[2, 1], [2, 3], [1, 3], [3, 1]], [1, 2], op)
    
    # EELS excitation
    # Width of electron beam and electron velocity
    width = 0.2
    vel = eelsbase.ene2vel(200e3)  # 200 keV electron energy
    
    # Impact parameters (triangle corner, middle, and midpoint of edge)
    imp = np.array([[-45, 0], [0, 0], [25, 0]])
    
    # Loss energies in eV
    ene = np.linspace(1.3, 4.3, 40)
    
    # Convert energies to nm
    unit_converter = units()
    enei = unit_converter.eV2nm / ene
    
    # BEM simulation
    # BEM solver
    bem = bemsolver(p, op)
    
    # Electron beam excitation
    exc = electronbeam(p, imp, width, vel, op)
    
    # Surface and bulk loss
    psurf = np.zeros((imp.shape[0], len(enei)))
    pbulk = np.zeros((imp.shape[0], len(enei)))
    
    print("Running BEM simulation for nanotriangle on membrane...")
    print("This may take several minutes due to complex geometry...")
    
    # Loop over energies
    for ien in tqdm(range(len(ene)), desc="BEM solver", ncols=80):
        # Surface charges
        sig = bem.solve(exc(enei[ien]))
        
        # EELS losses
        psurf[:, ien], pbulk[:, ien] = exc.loss(sig)
    
    print("Membrane EELS simulation completed!")
    
    # Total losses
    ptotal = psurf + pbulk
    
    # Final plot
    plt.figure(figsize=(12, 8))
    
    # Plot results
    impact_labels = ['Corner', 'Middle', 'Edge']
    colors = ['red', 'blue', 'green']
    
    for i in range(imp.shape[0]):
        plt.plot(ene, ptotal[i, :], color=colors[i], 
                label=impact_labels[i], linewidth=2, marker='o', markersize=4)
    
    plt.xlim([np.min(ene), np.max(ene)])
    plt.legend()
    plt.xlabel('Loss energy (eV)')
    plt.ylabel('Loss probability (eV⁻¹)')
    plt.title('EELS of silver nanotriangle on membrane (15 nm thick)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    analyze_membrane_effects(ene, ptotal, psurf, pbulk, imp)
    
    return ptotal, psurf, pbulk, ene, imp


def analyze_membrane_effects(ene, ptotal, psurf, pbulk, imp):
    """Analyze effects of membrane substrate on EELS"""
    
    print("\n=== Membrane Effects Analysis ===")
    
    impact_labels = ['Corner (-45, 0) nm', 'Middle (0, 0) nm', 'Edge (25, 0) nm']
    
    for i in range(len(impact_labels)):
        print(f"\n{impact_labels[i]}:")
        
        # Find peak characteristics
        peak_idx = np.argmax(ptotal[i, :])
        peak_energy = ene[peak_idx]
        peak_intensity = ptotal[i, peak_idx]
        
        print(f"  Peak energy: {peak_energy:.2f} eV")
        print(f"  Peak intensity: {peak_intensity:.3e} eV⁻¹")
        
        # Surface vs bulk contributions
        surf_total = np.sum(psurf[i, :])
        bulk_total = np.sum(pbulk[i, :])
        total = surf_total + bulk_total
        
        if total > 0:
            surf_fraction = surf_total / total
            bulk_fraction = bulk_total / total
            print(f"  Surface contribution: {surf_fraction*100:.1f}%")
            print(f"  Bulk contribution: {bulk_fraction*100:.1f}%")
        
        # Spectral width analysis
        half_max = peak_intensity / 2
        above_half = ene[ptotal[i, :] > half_max]
        if len(above_half) > 0:
            spectral_width = above_half[-1] - above_half[0]
            print(f"  Spectral width (FWHM): {spectral_width:.2f} eV")
    
    # Membrane substrate effects
    print(f"\nMembrane substrate effects:")
    print(f"- 15 nm thick dielectric membrane (ε = 4)")
    print(f"- Modified local field environment")
    print(f"- Additional interface plasmons possible")
    print(f"- Realistic TEM sample geometry")
    
    # Compare impact parameter responses
    max_intensities = [np.max(ptotal[i, :]) for i in range(3)]
    dominant_position = np.argmax(max_intensities)
    print(f"- Strongest response at: {impact_labels[dominant_position]}")


def plot_detailed_membrane_analysis(ene, ptotal, psurf, pbulk, imp):
    """Detailed analysis of membrane effects"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    impact_labels = ['Corner', 'Middle', 'Edge']
    colors = ['red', 'blue', 'green']
    
    # Total loss spectra
    for i in range(3):
        axes[0, 0].plot(ene, ptotal[i, :], color=colors[i], 
                       label=impact_labels[i], linewidth=2)
    axes[0, 0].set_xlabel('Loss energy (eV)')
    axes[0, 0].set_ylabel('Total loss probability (eV⁻¹)')
    axes[0, 0].set_title('Total EELS Response on Membrane')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Surface contributions
    for i in range(3):
        axes[0, 1].plot(ene, psurf[i, :], color=colors[i], 
                       label=f'{impact_labels[i]} (surf)', linewidth=2, linestyle='--')
    axes[0, 1].set_xlabel('Loss energy (eV)')
    axes[0, 1].set_ylabel('Surface loss probability (eV⁻¹)')
    axes[0, 1].set_title('Surface Contributions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bulk contributions
    for i in range(3):
        axes[1, 0].plot(ene, pbulk[i, :], color=colors[i], 
                       label=f'{impact_labels[i]} (bulk)', linewidth=2, linestyle=':')
    axes[1, 0].set_xlabel('Loss energy (eV)')
    axes[1, 0].set_ylabel('Bulk loss probability (eV⁻¹)')
    axes[1, 0].set_title('Bulk Contributions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Normalized comparison
    for i in range(3):
        ptotal_norm = ptotal[i, :] / np.max(ptotal[i, :])
        axes[1, 1].plot(ene, ptotal_norm, color=colors[i], 
                       label=impact_labels[i], linewidth=2)
    axes[1, 1].set_xlabel('Loss energy (eV)')
    axes[1, 1].set_ylabel('Normalized loss probability')
    axes[1, 1].set_title('Normalized EELS Spectra')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_membrane_geometry_effects(ene, ptotal):
    """Plot effects of membrane geometry on spectral response"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Spectral comparison
    impact_labels = ['Corner', 'Middle', 'Edge']
    colors = ['red', 'blue', 'green']
    
    axes[0].set_title('EELS Spectra on Membrane Substrate')
    for i in range(3):
        axes[0].plot(ene, ptotal[i, :], color=colors[i], 
                    label=impact_labels[i], linewidth=2)
    axes[0].set_xlabel('Loss energy (eV)')
    axes[0].set_ylabel('Loss probability (eV⁻¹)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Peak analysis
    peak_energies = []
    peak_intensities = []
    spectral_widths = []
    
    for i in range(3):
        peak_idx = np.argmax(ptotal[i, :])
        peak_energies.append(ene[peak_idx])
        peak_intensities.append(ptotal[i, peak_idx])
        
        # Calculate FWHM
        half_max = ptotal[i, peak_idx] / 2
        indices = np.where(ptotal[i, :] > half_max)[0]
        if len(indices) > 0:
            fwhm = ene[indices[-1]] - ene[indices[0]]
            spectral_widths.append(fwhm)
        else:
            spectral_widths.append(0)
    
    # Bar plot of characteristics
    x_pos = np.arange(3)
    width = 0.25
    
    bars1 = axes[1].bar(x_pos - width, peak_energies, width, 
                       label='Peak Energy (eV)', color='lightblue', alpha=0.7)
    bars2 = axes[1].bar(x_pos, [i*1000 for i in peak_intensities], width,
                       label='Peak Intensity (×1000 eV⁻¹)', color='lightgreen', alpha=0.7)
    bars3 = axes[1].bar(x_pos + width, spectral_widths, width,
                       label='FWHM (eV)', color='lightcoral', alpha=0.7)
    
    axes[1].set_xlabel('Impact Parameter')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Peak Characteristics')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(impact_labels)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (energy, intensity, width) in enumerate(zip(peak_energies, peak_intensities, spectral_widths)):
        axes[1].text(i-0.25, energy+0.05, f'{energy:.1f}', ha='center', va='bottom', fontsize=9)
        axes[1].text(i, intensity*1000+0.05, f'{intensity*1000:.0f}', ha='center', va='bottom', fontsize=9)
        axes[1].text(i+0.25, width+0.05, f'{width:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def explain_membrane_eels():
    """Explain membrane EELS physics"""
    
    print("\n=== Membrane EELS Physics ===")
    print("Membrane substrate effects:")
    print("- Realistic TEM sample geometry (suspended membrane)")
    print("- Modified boundary conditions vs infinite substrate")
    print("- Finite thickness creates additional resonances")
    print("- Interface plasmons at air-membrane boundaries")
    print("- Reduced substrate screening compared to bulk")
    print("- More complex field distributions")
    print("- Better represents experimental conditions")


if __name__ == '__main__':
    ptotal, psurf, pbulk, ene, imp = main()
    
    print("\n=== Simulation Summary ===")
    print(f"Silver nanotriangle: 80 nm base, 10 nm height")
    print(f"Membrane substrate: 15 nm thick, ε = 4")
    print(f"Electron beam: 200 keV, 0.2 nm width")
    print(f"Impact parameters: Corner (-45,0), Middle (0,0), Edge (25,0) nm")
    print(f"Energy range: {ene[0]:.1f} - {ene[-1]:.1f} eV")
    print(f"Complex geometry: Triangle + membrane with proper interfaces")
    
    # Detailed analysis
    plot_detailed_membrane_analysis(ene, ptotal, psurf, pbulk, imp)
    plot_membrane_geometry_effects(ene, ptotal)
    explain_membrane_eels()
    
    print(f"\nDemo/eels/ret 폴더의 모든 예제 변환이 완료되었습니다!")