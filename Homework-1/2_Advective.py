r"""
Solving an Advective PDE with finite differences.

The PDE described by

.. math::
    u_{t} + u_{x} = 0  \quad \forall x \in\Omega = [0, 1]  \;\; \& \;\;  t > 0

Whith a periodic boundary condition. The approximation used is a second order
finite difference scheme in space with both a forward and backward Euler method
of lines implementation to handle the time direction.

The goal is to implement the code in python and not rely on existing solvers.

Bram Lagerweij
COHMAS Mechanical Engineering KAUST
2021
"""

# Importing External modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Importing my own scripts
import sys
sys.path.insert(1, '../src')
from pde import advective
from time_integral import forwardEuler, backwardEuler

# Define properties
dx = 1e-2
dt = 1e-4
t_end = 1
c = 1  # Advective term

# Define discrete ranges
dof = int(1/dx) + 1
x = np.linspace(0, 1, dof)
dx = x[1] - x[0]
t = np.arange(0, t_end+dt, step=dt)

# Prepare solver
u = np.sin(2*np.pi*x)  # Initial condition

# Solve the problem using method of lines.
args = (dof, dx, c)
u_forw = forwardEuler(advective, u, dt, t_end, args=args)
u_back = backwardEuler(advective, u, dt, t_end, args=args)

# Plotting ploting statically
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_xlabel('$x$ location')
ax.set_ylabel('$u(x)$')

line_forw, = ax.plot(x, u_forw, label='forward')
line_back, = ax.plot(x, u_back, label='backward')
annotation = ax.annotate('time t=0', xy=(0.5, 1))

plt.legend()
