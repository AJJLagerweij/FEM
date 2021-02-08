r"""
Solving an Advective and Diffusive PDE with finite differences.

The PDE described by

.. math::
    u_{t} + u_{x} = \mu u_{xx}  \quad \forall x \in\Omega = [0, 1]  \;\; \& \;\;  t > 0

Whith a periodic boundary condition. It will show a combination of diffusive
and advective behaviour. The approximation used is a second order finite
difference scheme in space with both a forward and backward Euler method of
lines implementation to handle the time direction.

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
from pde import advectivediffusive
from time_integral import forwardEuler, backwardEuler


def plot_update(i):
    r"""
    Update the matplotlib plot.

    Parameters
    ----------
    i : int
        Iteration number.

    Returns
    -------
    line_forw : lines.line2D
        The updated Forward Euler predicition.
    line_back : lines.line2D
        The updated backwards Euler prediction.
    annotation : text.Annotation
        The annotated text shows the time progression.

    """
    line_forw.set_data(x, u_forw[i])
    line_back.set_data(x, u_back[i])
    annotation.set_text('time t={:.3f}'.format(dt*i))
    return line_forw, line_back, annotation


# Main code, hidden if this file is run as a module.
if __name__ == "__main__":
    # Define properties
    dx = 1e-2
    dt = 1e-4
    t_end = 1
    mu = 0.01  # Diffusive term
    c = 1  # Advective term

    # Define discrete ranges
    dof = int(1/dx) + 1
    x = np.linspace(0, 1, dof)
    dx = x[1] - x[0]
    t = np.arange(0, t_end+dt, step=dt)

    # Prepare solver
    u = np.sin(2*np.pi*x)  # Initial condition

    # Solve the problem using method of lines.
    u_forw = forwardEuler(advectivediffusive, u, dt, t_end, args=(dof, dx, mu, c))
    u_back = backwardEuler(advectivediffusive, u, dt, t_end, args=(dof, dx, mu, c))

    # Plotting with animation
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)

    line_forw, = ax.plot(x, u_forw[0], label='forward')
    line_back, = ax.plot(x, u_back[0], label='backward')
    annotation = ax.annotate('time t=0', xy=(0.5, 1))

    plt.legend()

    ani = animation.FuncAnimation(fig, plot_update, frames=len(u_forw),
                                  interval=1, blit=True, repeat=False)
