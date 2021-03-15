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

# Importing my own scripts
sys.path.insert(1, '../src')
from pde import projection, advectivediffusive
from finitedifference import advectivediffusive as advectivediffusive_fd
from solvers import solve, forwardEuler, backwardEuler
from element import interpolate
from helper import E1, E2


def mesh(x_start, x_end, n, order):
    """
    Meshing the 1D domain into `n` evenly spaced elements.

    Parameters
    ----------
    x_start : float
        Starting coordinate of the domain.
    x_end : float
        End coordinate of the domain.
    n : int
        Number of elements used to discretized the domain.
    order : int
        Order of the interpolation functions.

    Returns
    -------
    nodes : array_like(float), shape(n+1, order+1)
        For each node in each element the coordinates.
    connectivity array_like(int), shape(n+1, order+1)
        Elements to node connectivity array.
    """
    ele, dof = np.indices((n, order+1))
    connectivity = order*ele + dof
    ndofs = connectivity.max()+1
    nodes_x = np.linspace(x_start, x_end, ndofs)
    nodes = nodes_x[connectivity]

    # Now make it a periodic mesh by setting the last dof of the last element to 0.
    connectivity[-1, -1] = 0
    return nodes, connectivity


def exact(x, t, mu, c):
    r"""
    The exact solution to the pde:

    .. math::
        u(x, t) = \frac{3}{8} - \frac{1}{2} e^{-4\mu t} \cos(2(x-t))
                  + \frac{1}{8} e^{-16\mu t}\cos(4(x-t)

    Parameters
    ----------
    x : array_like(float)
        Locations :math:`x` where the function is evaluated.
    t : float
        Time of the solution.
    mu : float
        Amount of diffusivity.
    c : float
        Advection speed.

    Returns
    -------
    array_like(float)
        The function :math:`f(x)` at points `x`.
    """
    fun = 3/8 - 1/2 * np.exp(-4*mu*t) * np.cos(2*(x - c*t)) +\
          1/8 * np.exp(-16*mu*t) * np.cos(4*(x - c*t))
    return fun


@nb.jit(nopython=True)
def u0(x):
    r"""
    The initial condition as an exact function:

    .. math::
        f(x) = \sin^4(x)

    Parameters
    ----------
    x : array_like(float)
        Locations :math:`x` where the function is evaluated.

    Returns
    -------
    array_like(float)
        The function :math:`f(x)` at points `x`.
    """
    fun = np.sin(x) ** 4
    return fun


if __name__ == '__main__':
    # Define properties.
    num_q = 4  # number of quadrature points
    dt = 1e-4  # time step
    t_end = 0.3* 2*np.pi  # final time.
    mu = 0.01  # Diffusive term
    c = 1  # Advective term

    # Store error results.
    N_list = 2 * np.arange(1, 8)**2
    e1_forw_fd = []
    e2_forw_fd = []
    e1_back_fd = []
    e2_back_fd = []
    e1_forw_fe = []
    e2_forw_fe = []
    e1_back_fe = []
    e2_back_fe = []

    for N in N_list:
        # Because we are first order in time we need to refine time as well.
        # We scale spatially with: O(dx^2) and temporally with: O(dt^2).
        # Hence we set:
        dt = 1e-3 * (1/N)**2
        print(f'N={N}, h={1/N:1.2e}, dt={dt:1.2e}')

        # Solve using finite differences.
        dof = N + 1
        x_fd, dx = np.linspace(0, 2*np.pi, dof, retstep=True)
        t = np.arange(0, t_end + dt, step=dt)

        # Solve the problem using method of lines.
        u_forw_fd = forwardEuler(advectivediffusive_fd, u0(x_fd), dt, t_end, args=(dof, dx, mu, c))
        u_back_fd = backwardEuler(advectivediffusive_fd, u0(x_fd), dt, t_end, args=(dof, dx, mu, c))

        # Calculate errors.
        e1_forw_fd.append(E1(exact(x_fd, t_end, mu, c), u_forw_fd, x_fd))
        e2_forw_fd.append(E2(exact(x_fd, t_end, mu, c), u_forw_fd, x_fd))
        e1_back_fd.append(E1(exact(x_fd, t_end, mu, c), u_back_fd, x_fd))
        e2_back_fd.append(E2(exact(x_fd, t_end, mu, c), u_back_fd, x_fd))

        # Define mesh with linear elements.
        grid, connect = mesh(0, 2*np.pi, N, 1)

        # Prepare solver, for first time step we need to project the initial
        # condition onto our FE space.
        u = solve(projection, args=(grid, connect, u0, num_q, 1))

        # Solve the problem using method of lines.
        u_forw_fe = forwardEuler(advectivediffusive, u, dt, t_end, args=(grid, connect, c, mu, num_q, 1))
        u_back_fe = backwardEuler(advectivediffusive, u, dt, t_end, args=(grid, connect, c, mu, num_q, 1))

        # Interpolate the solutions at points x for plotting.
        x = np.linspace(0, 2 * np.pi, 10001)
        ux_forw_fe = interpolate(u_forw_fe, grid, connect, x, 1)
        ux_back_fe = interpolate(u_back_fe, grid, connect, x, 1)
        ux = interpolate(u, grid, connect, x, 1)

        # Calculate errors
        e1_forw_fe.append(E1(exact(x, t_end, mu, c), ux_forw_fe, x))
        e2_forw_fe.append(E2(exact(x, t_end, mu, c), ux_forw_fe, x))
        e1_back_fe.append(E1(exact(x, t_end, mu, c), ux_back_fe, x))
        e2_back_fe.append(E2(exact(x, t_end, mu, c), ux_back_fe, x))

        # Plot the distribution of the different methods.
        plt.figure(num='Solutions to the PDE')
        plt.xlim(0, 2 * np.pi)
        plt.ylim(0, 1)
        plt.xlabel('$x$ location')
        plt.ylabel('$u(x)$')
        plt.annotate(f'time t={t[-1]}\n# elements N={N}', xy=(0.1, 0.9))
        plt.tight_layout()
        plt.plot(x, exact(x, t_end, mu, c), label='Exact', zorder=1)
        plt.plot(x_fd, u_forw_fd, ':', label='FD forward')
        plt.plot(x_fd, u_back_fd, ':', label='FD backward')
        plt.plot(x, ux_forw_fe, ':', label='FE forward')
        plt.plot(x, ux_back_fe, ':', label='FE backward')
        plt.plot(x, u0(x), label='exact 0')
        plt.plot(x, ux, ':', label='u0')
        plt.legend(loc=1)
        plt.show()

    # Plot E1 with respect to the number of elements.
    plt.figure(num='E1 vs Elements')
    plt.plot(N_list, e1_forw_fd, 's', label='FD forward')
    plt.plot(N_list, e1_back_fd, 's', label='FD backward')
    plt.plot(N_list, e1_forw_fe, 's', label='FE forward')
    plt.plot(N_list, e1_back_fe, 's', label='FE backward')
    plt.plot(N_list, 1 / N_list ** 1, ':', label='$N^{-1}$')
    plt.plot(N_list, 1 / N_list ** 2, ':', label='$N^{-2}$')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('$E_2$')
    plt.xlabel('Number of Elements')
    plt.legend(loc=3)
    plt.tight_layout()

    # Plot E2 with respect to the number of elements.
    plt.figure(num='E2 vs Elements')
    plt.plot(N_list, e2_forw_fd, 's', label='FD forward')
    plt.plot(N_list, e2_back_fd, 's', label='FD backward')
    plt.plot(N_list, e2_forw_fe, 's', label='FE forward')
    plt.plot(N_list, e2_back_fe, 's', label='FE backward')
    plt.plot(N_list, 1 / N_list ** 1, ':', label='$N^{-1}$')
    plt.plot(N_list, 1 / N_list ** 2, ':', label='$N^{-2}$')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('$E_2$')
    plt.xlabel('Number of Elements')
    plt.legend(loc=3)
    plt.tight_layout()
