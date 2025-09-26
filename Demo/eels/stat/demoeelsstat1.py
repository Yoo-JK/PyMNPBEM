"""
DEMOEELSSTAT1 - Comparison BEM and Mie for EELS of metallic nanosphere.
For a silver nanosphere of 30 nm, this program computes the energy
loss probability for an impact parameter of 5 nm and for different
loss energies within the quasistatic approximation, and compares the
results with Mie theory.

See also F. J. Garcia de Abajo, Rev. Mod. Phys. 82, 209 (2010), Eq. (31).

Runtime on my computer: 5.4 seconds.
"""

import numpy as np
import matplotlib.pyplot as plt
from PyMNPBEM.base import bemoptions
from PyMNPBEM.material import epsconst, epstable
from PyMNPBEM.mesh2d import trisphere
from PyMNPBEM.particles import comparticle
from PyMNPBEM.bem import bemsolver
from PyMNPBEM.bem import electronbeam
from PyMNPBEM.base import eelsbase
from PyMNPBEM.mie import miesolver
from PyMNPBEM.misc import units, multiWaitbar

# Initialization
# Options for BEM simulation
op = bemoptions(sim='stat', interp='curv')

# Table of dielectric function
epstab = [epsconst(1), epstable('silver.dat')]

# Diameter
diameter = 30

# Nanosphere
p = comparticle(epstab, [trisphere(144, diameter)], [[2, 1]], 1, op)

# Width of electron beam and electron velocity
width, vel = 0.5, eelsbase.ene2vel(200e3)

# Impact parameter
imp = 5

# Loss energies in eV
ene = np.linspace(3, 4.5, 40)

# Convert energies to nm
units_obj = units()
enei = units_obj.eV2nm / ene

# BEM solution
# BEM solver
bem = bemsolver(p, op)

# Electron beam excitation
exc = electronbeam(p, [diameter / 2 + imp, 0], width, vel, op)

# Surface loss
psurf = np.zeros(ene.shape)

waitbar = multiWaitbar('BEM solver', 0, Color='g', CanCancel='on')

# Loop over wavelengths
for ien in range(len(enei)):
    # Surface charges
    sig = bem.solve(exc(enei[ien]))
    
    # EELS losses
    psurf[ien] = exc.loss(sig)
    
    waitbar.update('BEM solver', (ien + 1) / len(enei))

# Close waitbar
waitbar.close_all()

# Final figure and comparison with Mie solution
# Mie solver
mie = miesolver(epstab[1], epstab[0], diameter, op, lmax=40)

# Final plot
plt.figure()
plt.plot(ene, psurf, 'o-', label='BEM')
plt.plot(ene, mie.loss(imp, enei, vel), '.-', label='Mie')
plt.legend()
plt.xlabel('Loss energy (eV)')
plt.ylabel('Loss probability (eV^{-1})')
plt.title('EELS comparison: BEM vs Mie theory for Ag nanosphere')
plt.grid(True, alpha=0.3)
plt.show()