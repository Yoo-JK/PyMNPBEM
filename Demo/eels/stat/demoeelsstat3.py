"""
DEMOEELSSTAT3 - EELS of nanodisk for different impact parameters.
For a silver nanodisk with 30 nm diameter and 5 nm height, this
program computes the energy loss probability for different impact
parameters and loss energies within the quasistatic approximation.

Runtime on my computer: 70 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from PyMNPBEM.base import bemoptions
from PyMNPBEM.material import epsconst, epstable
from PyMNPBEM.mesh2d import polygon, edgeprofile, tripolygon
from PyMNPBEM.particles import comparticle
from PyMNPBEM.bem import bemsolver
from PyMNPBEM.bem import electronbeam
from PyMNPBEM.base import eelsbase
from PyMNPBEM.misc import units, multiWaitbar

# Initialization
# Options for BEM simulation
op = bemoptions(sim='stat', interp='curv')

# Table of dielectric function
epstab = [epsconst(1), epstable('silver.dat')]

# Diameter of disk
diameter = 30

# Polygon for disk
poly = polygon(25, size=[1, 1] * diameter)

# Edge profile for disk
edge = edgeprofile(5, 11)

# Extrude polygon to nanoparticle
p = comparticle(epstab, [tripolygon(poly, edge)], [[2, 1]], 1, op)

# Width of electron beam and electron velocity
width, vel = 0.5, eelsbase.ene2vel(200e3)

# Impact parameters
imp = np.linspace(0, 1.4 * diameter, 81)

# Loss energies in eV
ene = np.linspace(2.5, 4.5, 60)

# Convert energies to nm
units_obj = units()
enei = units_obj.eV2nm / ene

# BEM solution
# BEM solver
bem = bemsolver(p, op)

# Electron beam excitation - convert impact parameters to 2D positions
imp_positions = np.column_stack([imp, np.zeros(len(imp))])
exc = electronbeam(p, imp_positions, width, vel, op)

# Surface and bulk losses
psurf = np.zeros((len(imp), len(enei)))
pbulk = np.zeros((len(imp), len(enei)))

waitbar = multiWaitbar('BEM solver', 0, Color='g', CanCancel='on')

# Loop over wavelengths
for ien in range(len(enei)):
    # Surface charges
    sig = bem.solve(exc(enei[ien]))
    
    # EELS losses
    psurf[:, ien], pbulk[:, ien] = exc.loss(sig)
    
    waitbar.update('BEM solver', (ien + 1) / len(enei))

# Close waitbar
waitbar.close_all()

# Final plot
# Electron energy loss probability
plt.figure(figsize=(10, 8))
plt.imshow(psurf + pbulk, 
           extent=[ene[0], ene[-1], imp[0], imp[-1]], 
           aspect='auto', 
           origin='lower',
           cmap='viridis')

# Plot disk edge
plt.plot(ene, np.full_like(ene, diameter / 2), 'w--', linewidth=2, label='Disk edge')

plt.colorbar(label='Loss probability (eV⁻¹)')
plt.xlabel('Loss energy (eV)')
plt.ylabel('Impact parameter (nm)')
plt.title('EELS Loss probability for Ag nanodisk (30nm diameter, 5nm height)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()