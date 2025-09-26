"""
DEMOEELSSTAT2 - EELS of nanosphere for different impact parameters.
For a silver nanosphere with 30 nm diameter, this program computes the
energy loss probability for different parameters and loss energies
within the quasistatic approximation.

Runtime on my computer: 82 seconds.
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
from PyMNPBEM.misc import units, multiWaitbar

# Initialization
# Options for BEM simulation
op = bemoptions(sim='stat', interp='curv')

# Table of dielectric function
epstab = [epsconst(1), epstable('silver.dat')]

# Diameter
diameter = 30

# Nanosphere
p = comparticle(epstab, [trisphere(256, diameter)], [[2, 1]], 1, op)

# Width of electron beam and electron velocity
width, vel = 0.5, eelsbase.ene2vel(200e3)

# Impact parameters
imp = np.linspace(0, 40, 81)

# Loss energies in eV
ene = np.linspace(3, 4.5, 60)

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
# Density plot of total scattering rate (LDOS)
plt.figure(figsize=(10, 8))
plt.imshow(psurf + pbulk, 
           extent=[ene[0], ene[-1], imp[0], imp[-1]], 
           aspect='auto', 
           origin='lower',
           cmap='viridis')

# Plot disk edge
plt.plot(ene, np.full_like(ene, 15), 'w--', linewidth=2, label='Sphere edge')

plt.colorbar(label='Loss probability (eV⁻¹)')
plt.xlabel('Loss energy (eV)')
plt.ylabel('Impact parameter (nm)')
plt.title('EELS Loss probability for Ag nanosphere')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()