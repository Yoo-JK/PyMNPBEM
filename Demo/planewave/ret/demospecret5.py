"""
DEMOSPECRET5 - Field enhancement for Au nanosphere in Ag nanocube.
For a Au nanosphere (diameter 35 nm) in a Ag nanocube (size 100 nm)
and an incoming plane wave with a polarization along x, this program
first computes the surface charge at the resonance wavelength of 470
nm, and then the electric fields outside and inside the nanoparticle.
For an experimental realization of the nanoparticle see
G. Boris et al., J. Chem. Phys. C 118, 15356 (2014).

Runtime on my computer: 53 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from PyMNPBEM.base import bemoptions
from PyMNPBEM.material import epsconst, epstable
from PyMNPBEM.mesh2d import trisphere, tricube
from PyMNPBEM.particles import comparticle
from PyMNPBEM.bem import bemsolver
from PyMNPBEM.bem import planewave
from PyMNPBEM.bem import meshfield

# Initialization
# Options for BEM simulation
op = bemoptions(sim='ret', interp='curv', refine=2)

# Table of dielectric functions
epstab = [epsconst(1), epstable('silver.dat'), epstable('gold.dat')]

# Nanosphere (Au, diameter 70 nm)
p1 = trisphere(144, 70)

# Rounded nanocube (Ag, size 100 nm)
p2 = tricube(12, 100)

# Initialize gold sphere in silver cube
p = comparticle(epstab, [p1, p2], [[3, 2], [2, 1]], 1, 2, op)

# BEM simulation
# Set up BEM solver
bem = bemsolver(p, op)

# Plane wave excitation (x-polarization)
exc = planewave([1, 0, 0], [0, 0, 1], op)

# Light wavelength in vacuum (resonance wavelength)
enei = 470

# Surface charge
sig = bem.solve(exc(p, enei))

# Computation of electric field
# Mesh for calculation of electric field
x, z = np.meshgrid(np.linspace(-70, 70, 81), np.linspace(-70, 70, 81))
y = np.zeros_like(x)  # y = 0 plane (xz cross-section)

# Object for electric field
# MINDIST controls the minimal distance of the field points to the particle boundary
emesh = meshfield(p, x, y, z, op, mindist=0.9, nmax=2000)

# Induced and incoming electric field
e_induced = emesh(sig)
e_incoming = emesh(exc.field(emesh.pt, enei))
e_total = e_induced + e_incoming

# Norm of electric field
ee = np.sqrt(np.sum(np.abs(e_total)**2, axis=-1))

# Final plot
plt.figure(figsize=(12, 10))

# Main plot: Electric field enhancement
plt.subplot(2, 2, 1)
im1 = plt.imshow(ee, extent=[-70, 70, -70, 70], origin='lower', cmap='hot')
plt.colorbar(im1, label='|E|/|E₀|')
plt.xlabel('x (nm)')
plt.ylabel('z (nm)')
plt.title('Electric Field Enhancement at λ = 470 nm')
plt.axis('equal')

# Add particle boundaries (approximate)
circle1 = plt.Circle((0, 0), 35, fill=False, color='white', linewidth=2, linestyle='--', label='Au sphere')
square = plt.Rectangle((-50, -50), 100, 100, fill=False, color='cyan', linewidth=2, linestyle='-', label='Ag cube')
plt.gca().add_patch(circle1)
plt.gca().add_patch(square)
plt.legend()

# Cross-section plots
plt.subplot(2, 2, 2)
center_idx = len(x) // 2
plt.plot(x[center_idx, :], ee[center_idx, :], 'r-', linewidth=2)
plt.xlabel('x (nm)')
plt.ylabel('|E|/|E₀|')
plt.title('Field Enhancement along x-axis (z=0)')
plt.grid(True, alpha=0.3)
plt.axvline(-35, color='gold', linestyle='--', alpha=0.7, label='Au sphere edge')
plt.axvline(35, color='gold', linestyle='--', alpha=0.7)
plt.axvline(-50, color='silver', linestyle='-', alpha=0.7, label='Ag cube edge')
plt.axvline(50, color='silver', linestyle='-', alpha=0.7)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(z[:, center_idx], ee[:, center_idx], 'b-', linewidth=2)
plt.xlabel('z (nm)')
plt.ylabel('|E|/|E₀|')
plt.title('Field Enhancement along z-axis (x=0)')
plt.grid(True, alpha=0.3)
plt.axvline(-35, color='gold', linestyle='--', alpha=0.7, label='Au sphere edge')
plt.axvline(35, color='gold', linestyle='--', alpha=0.7)
plt.axvline(-50, color='silver', linestyle='-', alpha=0.7, label='Ag cube edge')
plt.axvline(50, color='silver', linestyle='-', alpha=0.7)
plt.legend()

# Log scale plot
plt.subplot(2, 2, 4)
im2 = plt.imshow(np.log10(ee + 1e-10), extent=[-70, 70, -70, 70], origin='lower', cmap='viridis')
plt.colorbar(im2, label='log₁₀(|E|/|E₀|)')
plt.xlabel('x (nm)')
plt.ylabel('z (nm)')
plt.title('Electric Field Enhancement (log scale)')
plt.axis('equal')

# Add particle boundaries
circle2 = plt.Circle((0, 0), 35, fill=False, color='white', linewidth=2, linestyle='--')
square2 = plt.Rectangle((-50, -50), 100, 100, fill=False, color='cyan', linewidth=2, linestyle='-')
plt.gca().add_patch(circle2)
plt.gca().add_patch(square2)

plt.tight_layout()
plt.show()

# Print some analysis
max_enhancement = np.max(ee)
print(f"Maximum field enhancement: {max_enhancement:.1f}")
print(f"Structure: Au nanosphere (35nm radius) in Ag nanocube (100nm)")
print(f"Resonance wavelength: {enei} nm")
print("Reference: G. Boris et al., J. Chem. Phys. C 118, 15356 (2014)")