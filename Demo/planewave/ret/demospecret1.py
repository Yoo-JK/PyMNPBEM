"""
DEMOSPECRET1 - Light scattering of metallic nanosphere.
For a metallic nanosphere and an incoming plane wave, this program
computes the scattering cross section for different light wavelengths
using the full Maxwell equations, and compares the results with Mie
theory.

Runtime on my computer: 7.4 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from PyMNPBEM.base import bemoptions
from PyMNPBEM.material import epsconst, epstable
from PyMNPBEM.mesh2d import trisphere
from PyMNPBEM.particles import comparticle
from PyMNPBEM.bem import bemsolver
from PyMNPBEM.bem import planewave
from PyMNPBEM.mie import miesolver
from PyMNPBEM.misc import multiWaitbar

# Initialization
# Options for BEM simulation
op = bemoptions(sim='ret', interp='curv')

# Table of dielectric functions
epstab = [epsconst(1), epstable('gold.dat')]

# Diameter of sphere
diameter = 150

# Initialize sphere
p = comparticle(epstab, [trisphere(144, diameter)], [[2, 1]], 1, op)

# BEM simulation
# Set up BEM solver
bem = bemsolver(p, op)

# Plane wave excitation - define polarization directions and propagation
# [1, 0, 0; 0, 1, 0] for x and y polarizations
# [0, 0, 1; 0, 0, 1] for z-direction propagation
pol_directions = np.array([[1, 0, 0], [0, 1, 0]])  # x and y polarizations
prop_directions = np.array([[0, 0, 1], [0, 0, 1]])  # z-direction propagation
exc = planewave(pol_directions, prop_directions, op)

# Light wavelength in vacuum
enei = np.linspace(400, 900, 40)

# Allocate scattering and extinction cross sections
sca = np.zeros((len(enei), 2))  # 2 polarizations
ext = np.zeros((len(enei), 2))

waitbar = multiWaitbar('BEM solver', 0, Color='g', CanCancel='on')

# Loop over wavelengths
for ien in range(len(enei)):
    # Surface charge
    sig = bem.solve(exc(p, enei[ien]))
    
    # Scattering and extinction cross sections
    sca[ien, :] = exc.sca(sig)
    ext[ien, :] = exc.ext(sig)
    
    waitbar.update('BEM solver', (ien + 1) / len(enei))

# Close waitbar
waitbar.close_all()

# Final plot
plt.figure(figsize=(10, 6))
plt.plot(enei, sca[:, 0], 'o-', label='BEM : x-polarization')
plt.plot(enei, sca[:, 1], 's-', label='BEM : y-polarization')

plt.xlabel('Wavelength (nm)')
plt.ylabel('Scattering cross section (nm²)')

# Comparison with Mie theory
mie = miesolver(epstab[1], epstab[0], diameter, op)
mie_sca = mie.sca(enei)
plt.plot(enei, mie_sca, '--', linewidth=2, label='Mie theory')

plt.legend()
plt.title('Light Scattering: BEM vs Mie Theory for Au Nanosphere (150nm)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()