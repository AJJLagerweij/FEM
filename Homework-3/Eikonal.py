r"""
Solving an Advective and Diffusive PDE with finite elements.

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
import numba as nb
import numpy as np
from scipy.spatial import Delaunay

# Importing my own scripts
sys.path.insert(1, '../src')
from pde import projection, advectivediffusive
from solvers import solve, forwardEuler, backwardEuler
# from element import interpolate
# from helper import E1, E2


def mesh(x_start, x_end, n):
    """
    Meshing the 1D domain into `n` evenly spaced elements.

    Parameters
    ----------
    x_start : list
        Starting coordinate of the domain [x1,x2].
    x_end : list
        End coordinate of the domain [x1,x2].
    n : int
        Number of elements used to discretized the domain.

    Returns
    -------
    nodes : array_like(float)
        Coordinates of all the nodal points.
    connectivity array_like(int), shape((n+1, order+1))
        Elements to node connectivity array.
    """
    x = np.linspace(x_start[0], x_end[1], n+1)
    y = np.linspace(x_start[1], x_end[0], n+1)
    xx, yy = np.meshgrid(x, y)
    points = np.c_[xx.reshape(-1), yy.reshape(-1)]
    mesh = Delaunay(points)
    return points, mesh.vertices


@nb.jit(nopython=True)
def u0(x, y):
    r"""
    The initial condition as an exact function:

    .. math::
        f(x, y) = begin{cases} 1 & (x-0.5)^2 + (y-0.5)^2 \leq 0.25^2 \\ -1 & \text{otherwise} \end{cases}

    Parameters
    ----------
    x : array_like(float)
        Locations :math:`x` where the function is evaluated.
    y : array_like(float)
        Locations :math:`y` where the function is evaluated.

    Returns
    -------
    array_like(float)
        The function :math:`f(x,y)` at points `x` and `y`.
    """
    fun = 1*((x-0.5)**2 + (y-0.5)**2 <= 0.25**2) - 1*((x-0.5)**2 + (y-0.5)**2 > 0.25**2)
    return fun


# if __name__ == '__main__':
# Define properties.
N = 10  # number of elements.
num_q = 4  # number of quadrature points
dt = 1e-4  # time step
t_end = 2 * np.pi  # final time.
mu = 0.01  # Diffusive term
c = 1  # Advective term

# Define mesh with linear elements.
grid, connect = mesh([0, 0], [1, 1], N)


# t = np.arange(0, t_end + dt, step=dt)

# Prepare solver, for first time step we need to project the initial
# condition onto our FE space.
u = solve(projection, args=(grid, connect, u0, num_q, 1))
#
# plt.plot(grid, )

    # Solve the problem using method of lines.
#     u_forw = forwardEuler(advectivediffusive, u0, dt, t_end, args=(grid, connect, mu, c, num_q, 1))
#     u_back = backwardEuler(advectivediffusive, u0, dt, t_end, args=(grid, connect, mu, c, num_q, 1))
#
#     # Plotting the results.
#     plt.xlim(0, 1)
#     plt.ylim(-1, 1)
#     plt.xlabel('$x$ location')
#     plt.ylabel('$u(x)$')
#     plt.annotate('time t={}'.format(t[-1]), xy=(0.5, 0.9), ha='center')
#     plt.tight_layout()
#
#     plt.plot(x, u_forw, label='forward')
#     plt.plot(x, u_back, label='backward')
#
#     plt.legend()
#     plt.show()