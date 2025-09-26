"""
DEMOSPECRET4 - Spectrum for Au nanosphere in Ag nanocube.
For a Au nanosphere (diameter 35 nm) in a Ag nanocube (size 100 nm)
and an incoming plane wave with a polarization along x or y, this
program computes the scattering cross section for different light wavelengths
using the full Maxwell equations. For an experimental realization see
G. Boris et al., J. Chem. Phys. C 118, 15356 (2014).

Runtime on my computer: 112 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from PyMNPBEM.base import bemoptions
from PyMNPBEM.material import epsconst, epstable
from PyMNPBEM.mesh2d import trisphere, tricube
from PyMNPBEM.particles import comparticle
from PyMNPBEM.bem import bemsolver
from PyMNPBEM.bem import planewave
from PyMNPBEM.misc import multiWaitbar

# Initialization
# Options for BEM simulation
op = bemoptions(sim='ret', interp='curv')

# Table of dielectric functions
epstab = [epsconst(1), epstable('silver.dat'), epstable('gold.dat')]

# Nanosphere (Au, diameter 70 nm -> radius 35 nm)
p1 = trisphere(144, 70)

# Rounded nanocube (Ag, size 100 nm)
p2 = tricube(12, 100)

# Initialize gold sphere in silver cube
# [3, 2; 2, 1] means:
# - Inner sphere (p1): gold (index 3) inside silver (index 2)  
# - Outer cube (p2): silver (index 2) inside air/vacuum (index 1)
# Medium 1 (air), Medium 2 (silver cube)
p = comparticle(epstab, [p1, p2], [[3, 2], [2, 1]], 1, 2, op)

# BEM simulation
# Set up BEM solver
bem = bemsolver(p, op)

# Plane wave excitation
pol_directions = np.array([[1, 0, 0], [0, 1, 0]])  # x and y polarizations
prop_directions = np.array([[0, 0, 1], [0, 0, 1]])  # z-direction propagation
exc = planewave(pol_directions, prop_directions, op)

# Light wavelength in vacuum
enei = np.linspace(300, 750, 40)

# Allocate scattering cross sections
sca = np.zeros((len(enei), 2))

waitbar = multiWaitbar('BEM solver', 0, Color='g', CanCancel='on')

# Loop over wavelengths
for ien in range(len(enei)):
    # Surface charge
    sig = bem.solve(exc(p, enei[ien]))
    
    # Scattering cross sections
    sca[ien, :] = exc.sca(sig)
    
    waitbar.update('BEM solver', (ien + 1) / len(enei))

# Close waitbar
waitbar.close_all()

# Final plot
plt.figure(figsize=(10, 6))
plt.plot(enei, sca[:, 0], 'o-', label='x-pol', markersize=4)
plt.plot(enei, sca[:, 1], 's-', label='y-pol', markersize=4)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Scattering cross section (nm²)')
plt.title('Scattering Spectrum: Au Nanosphere (35nm) in Ag Nanocube (100nm)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print some information about the structure
print("Structure: Au nanosphere (diameter 70nm) inside Ag nanocube (size 100nm)")
print("Materials: Air (ε=1), Silver, Gold")
print("Calculation: Full Maxwell equations with retardation")
print(f"Wavelength range: {enei[0]:.0f} - {enei[-1]:.0f} nm")
print("Reference: G. Boris et al., J. Chem. Phys. C 118, 15356 (2014)")