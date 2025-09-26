"""
DEMOSPECRET3 - Field enhancement for nanodisk.
For a metallic nanodisk and an incoming plane wave with a polarization
along x, this program computes and plots the field enhancement using
the full Maxwell equations. We show how to plot fields on the
particle boundaries and how to set up a plate around the nanodisk,
using the POLYGON3 class, on which we plot the electric fields.

Runtime on my computer: 20.7 sec.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PyMNPBEM.base import bemoptions
from PyMNPBEM.material import epsconst, epstable
from PyMNPBEM.mesh2d import polygon, edgeprofile, tripolygon, polygon3, plate, shiftbnd
from PyMNPBEM.particles import comparticle, compoint
from PyMNPBEM.bem import bemsolver
from PyMNPBEM.bem import planewave
from PyMNPBEM.greenfun import greenfunction
from PyMNPBEM.bem import field
from PyMNPBEM.misc import vecnorm, vecnormalize

# Initializations
# Options for BEM simulation
op = bemoptions(sim='ret', interp='curv')

# Table of dielectric functions
epstab = [epsconst(1), epstable('gold.dat')]

# Polygon for disk
poly = polygon(30, size=[30, 30])

# Edge profile for nanodisk
edge = edgeprofile(5, 11)

# Extrude polygon to nanoparticle
p_geometry, poly = tripolygon(poly, edge)

# Initialize particle
p = comparticle(epstab, [p_geometry], [[2, 1]], 1, op)

# BEM simulation and plot of electric field on particle
# Set up BEM solver
bem = bemsolver(p, op)

# Plane wave excitation
exc = planewave([1, 0, 0], [0, 0, 1], op)

# Light wavelength in vacuum
enei = 580

# Surface charge
sig = bem.solve(exc(p, enei))

# Electric field
f = field(bem, sig)

# Plot norm of induced electric field on particle surface
fig = plt.figure(figsize=(15, 10))

# First subplot: Electric field on particle surface
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
p.plot(ax1, field_data=vecnorm(f.e), colorbar=True)
ax1.set_title('Electric field enhancement on particle surface')

# Plot electric field around particle
# Change z-value polygon returned by TRIPOLYGON and shift boundaries to outside
poly1 = shiftbnd(poly.set_z(-2), 1)

# Change direction of polygon so that it becomes the inner plate boundary
poly1 = poly1.set_dir(-1)

# Outer plate boundary
poly2 = polygon3(polygon(4, size=[60, 60]), -2)

# Make plate
eplate = plate([poly1, poly2])

# Make COMPOINT object with plate positions
pt1 = compoint(p, eplate.verts)

# Set up Green function object between PT and P
g1 = greenfunction(pt1, p, op)

# Induced electric field at plate vertices
fplate = field(g1, sig)

# Second subplot: Electric field on plate
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
eplate.plot(ax2, field_data=vecnorm(fplate.e), colorbar=True)
ax2.set_title('Electric field around particle')

# Overlay with quiver plot
# Grid for field vectors
x, y = np.meshgrid(np.linspace(-30, 30, 31), np.linspace(-30, 30, 31))
z = np.full_like(x, -2)

# Make COMPOINT object with grid positions
grid_pos = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
pt2 = compoint(p, grid_pos, op, mindist=2, medium=1)

# Set up Green function
g2 = greenfunction(pt2, p, op)

# Electric field at grid positions
f2 = field(g2, sig)

# Normalized electric field
e2 = np.imag(vecnormalize(f2.e))

# Third subplot: Quiver plot of electric field vectors
ax3 = fig.add_subplot(2, 2, 3, projection='3d')

# Plot particle boundary
p.plot(ax3, alpha=0.7)

# Add quiver plot
ax3.quiver(pt2.pos[:, 0], pt2.pos[:, 1], pt2.pos[:, 2],
           e2[:, 0], e2[:, 1], e2[:, 2], 
           length=0.5, color='white', alpha=0.8)

ax3.set_title('Electric field vectors around particle')
ax3.set_xlabel('X (nm)')
ax3.set_ylabel('Y (nm)')
ax3.set_zlabel('Z (nm)')

# Fourth subplot: 2D cross-section view
ax4 = fig.add_subplot(2, 2, 4)

# Create 2D cross-section at z = -2
field_magnitude = vecnorm(f2.e).reshape(x.shape)
contour = ax4.contourf(x, y, field_magnitude, levels=20, cmap='viridis')
plt.colorbar(contour, ax=ax4, label='|E|/|E₀|')

# Add quiver plot for 2D view
step = 3  # Show every 3rd vector for clarity
ax4.quiver(x[::step, ::step], y[::step, ::step], 
           e2.reshape(x.shape)[::step, ::step, 0], 
           e2.reshape(x.shape)[::step, ::step, 1],
           color='white', alpha=0.7, scale=20)

# Add particle outline
particle_circle = plt.Circle((0, 0), 15, fill=False, color='white', linewidth=2)
ax4.add_patch(particle_circle)

ax4.set_xlim(-30, 30)
ax4.set_ylim(-30, 30)
ax4.set_xlabel('X (nm)')
ax4.set_ylabel('Y (nm)')
ax4.set_title('2D cross-section: Electric field enhancement')
ax4.set_aspect('equal')

plt.tight_layout()
plt.show()