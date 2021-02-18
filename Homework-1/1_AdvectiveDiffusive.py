r"""
Solving an Advective and Diffusive PDE with finite differences.

The PDE described by

.. math::
    u_{t} + u_{x} = \mu u_{xx}  \quad \forall x \in\Omega = [0, 1]  \;\; \& \;\;  t > 0

With a periodic boundary condition. It will show a combination of diffusive
and advective behaviour. The approximation used is a second order finite
difference scheme in space with both a forward and backward Euler method of
lines implementation to handle the time direction.

The goal is to implement the code in python and not rely on existing solvers.

Bram Lagerweij
COHMAS Mechanical Engineering KAUST
2021
"""

# Importing External modules
import sys
import matplotlib.pyplot as plt
import numpy as np

# Importing my own scripts
sys.path.insert(1, '../src')
from pde import advectivediffusive
from time_integral import forwardEuler, backwardEuler


if __name__ == '__main__':
    # Define properties
    dx = 1e-2
    dt = 1e-4
    t_end = 1
    mu = 0.01  # Diffusive term
    c = 1  # Advective term

    # Define discrete ranges
    dof = int(1 / dx) + 1
    x, dx = np.linspace(0, 1, dof, retstep=True)
    t = np.arange(0, t_end + dt, step=dt)

    # Prepare solver
    u0 = np.sin(2 * np.pi * x)  # Initial condition

    # Solve the problem using method of lines.
    u_forw = forwardEuler(advectivediffusive, u0, dt, t_end, args=(dof, dx, mu, c))
    u_back = backwardEuler(advectivediffusive, u0, dt, t_end, args=(dof, dx, mu, c))

    # Plotting plotting statically
    plt.xlim(0, 1)
    plt.ylim(-1, 1)
    plt.xlabel('$x$ location')
    plt.ylabel('$u(x)$')
    plt.annotate('time t={}'.format(t[-1]), xy=(0.5, 0.9), ha='center')
    plt.tight_layout()

    plt.plot(x, u_forw, label='forward')
    plt.plot(x, u_back, label='backward')

    plt.legend()
    plt.show()
