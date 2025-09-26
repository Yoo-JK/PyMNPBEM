"""
DEMOSPECRET2 - Light scattering of metallic nanodisk.
For a metallic nanodisk and an incoming plane wave, this program
computes the scattering cross section for different light wavelengths
using the full Maxwell equations, and compares the results with those
obtained within the quasistatic approximation.

Runtime on my computer: 18 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from PyMNPBEM.base import bemoptions
from PyMNPBEM.material import epsconst, epstable
from PyMNPBEM.mesh2d import polygon, edgeprofile, tripolygon
from PyMNPBEM.particles import comparticle
from PyMNPBEM.bem import bemsolver
from PyMNPBEM.bem import planewave
from PyMNPBEM.misc import multiWaitbar

# Initialization
# Options for BEM simulation
op1 = bemoptions(sim='stat', interp='curv')  # Quasistatic approximation
op2 = bemoptions(sim='ret', interp='curv')   # Full retarded calculation

# Table of dielectric functions
epstab = [epsconst(1), epstable('gold.dat')]

# Polygon for disk
poly = polygon(25, size=[30, 30])

# Edge profile for nanodisk
edge = edgeprofile(5, 11)

# Extrude polygon to nanoparticle
p_geometry = tripolygon(poly, edge)

# Initialize particles for both calculations
p = comparticle(epstab, [p_geometry], [[2, 1]], 1, op1)

# BEM simulation
# Set up BEM solvers
bem1 = bemsolver(p, op1)  # Static solver
bem2 = bemsolver(p, op2)  # Retarded solver

# Plane wave excitation
pol_directions = np.array([[1, 0, 0], [0, 1, 0]])  # x and y polarizations
prop_directions = np.array([[0, 0, 1], [0, 0, 1]])  # z-direction propagation
exc1 = planewave(pol_directions, prop_directions, op1)
exc2 = planewave(pol_directions, prop_directions, op2)

# Light wavelength in vacuum
enei = np.linspace(450, 750, 40)

# Allocate scattering cross sections
sca1 = np.zeros((len(enei), 2))  # Static approximation
sca2 = np.zeros((len(enei), 2))  # Retarded calculation

waitbar = multiWaitbar('BEM solver', 0, Color='g', CanCancel='on')

# Loop over wavelengths
for ien in range(len(enei)):
    # Surface charge
    sig1 = bem1.solve(exc1(p, enei[ien]))
    sig2 = bem2.solve(exc2(p, enei[ien]))
    
    # Scattering cross sections
    sca1[ien, :] = exc1.sca(sig1)
    sca2[ien, :] = exc2.sca(sig2)
    
    waitbar.update('BEM solver', (ien + 1) / len(enei))

# Close waitbar
waitbar.close_all()

# Final plot
plt.figure(figsize=(12, 8))

# Plot static approximation results
plt.plot(enei, sca1[:, 0], 'o--', label='x-pol, static', markersize=4)
plt.plot(enei, sca1[:, 1], 's--', label='y-pol, static', markersize=4)

# Plot retarded calculation results
plt.plot(enei, sca2[:, 0], 'o-', label='x-pol, retarded', markersize=4)
plt.plot(enei, sca2[:, 1], 's-', label='y-pol, retarded', markersize=4)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Scattering cross section (nm²)')
plt.title('Light Scattering: Static vs Retarded for Au Nanodisk (30nm diameter, 5nm height)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()