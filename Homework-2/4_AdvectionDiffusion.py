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
import numpy as np
import numba as nb
nb.NUMBA_DISABLE_JIT = 0  # Turn JIT on (0) and off (1)

# Importing my own scripts
sys.path.insert(1, '../src')
from pde import projection, advectivediffusive
from finitedifference import advectivediffusive as advectivediffusive_fd
from solvers import solve, forwardEuler, backwardEuler
from element import Mesh1D
from fem import interpolate
from helper import E1, E2


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
    t_end = 2*np.pi  # final time.
    mu = 0.01  # Diffusive term
    c = 1  # Advective term

    # Store error results.
    N_list = 2**np.arange(2, 10)
    x = np.linspace(0, 2 * np.pi, 10001)  # Locations for error
    e1_forw_fd = []
    e2_forw_fd = []
    e1_back_fd = []
    e2_back_fd = []
    e1_forw_fe = []
    e2_forw_fe = []
    e1_back_fe = []
    e2_back_fe = []

    # Initialize plotting.
    plt.figure(num='Solutions to the PDE')
    plt.xlim(0, 2 * np.pi)
    plt.ylim(0, 1)
    plt.xlabel('$x$ location')
    plt.ylabel('$u(x)$')
    plt.annotate(f"time t={t_end:1.3}s", xy=(0.1, 0.95))
    plt.plot(x, exact(x, t_end, mu, c), label='Exact', zorder=1)
    plt.tight_layout()

    # Solve the different
    for N in N_list:
        # Because we are first order in time we need to refine time as well.
        # We scale spatially with: O(dx^2) and temporally with: O(dt^2).
        # Hence we set:
        dt = 0.1 * (1/N)**2
        print(f'N={N}, h={1/N:1.2e}, dt={dt:1.2e}')

        # Solve using finite differences.
        dof = N + 1
        x_fd, dx = np.linspace(0, 2*np.pi, dof, retstep=True)
        t = np.arange(0, t_end + dt, step=dt)

        # Solve the problem using method of lines.
        pde_fd = advectivediffusive_fd(dof, dx, mu, c)  # Setup
        u_forw_fd = forwardEuler(pde_fd, u0(x_fd), dt, t_end)  # Solve
        u_back_fd = backwardEuler(pde_fd, u0(x_fd), dt, t_end)

        # Calculate errors.
        reference_fd = exact(x_fd, t_end, mu, c)
        e1_forw_fd.append(E1(reference_fd, u_forw_fd, x_fd))
        e2_forw_fd.append(E2(reference_fd, u_forw_fd, x_fd))
        e1_back_fd.append(E1(reference_fd, u_back_fd, x_fd))
        e2_back_fd.append(E2(reference_fd, u_back_fd, x_fd))

        # Define mesh with linear elements.
        mesh = Mesh1D(0.0, 2*np.pi, N, 1, num_q, periodic=True)  # Setup

        # Prepare solver, for first time step we need to project the initial
        # condition onto our FE space.
        project = projection(mesh, u0)  # Assemble
        u = solve(*project)  # Solve

        # Solve the problem using method of lines.
        pde = advectivediffusive(mesh, c, mu)  # Assemble
        u_forw_fe = forwardEuler(pde, u, dt, t_end)  # Solve
        u_back_fe = backwardEuler(pde, u, dt, t_end)  # Solve

        # Interpolate the solutions at points x for plotting.
        ux_forw_fe = interpolate(mesh, u_forw_fe, x)  # Post Process
        ux_back_fe = interpolate(mesh, u_back_fe, x)  # Post Process

        # Calculate errors
        reference_fe = exact(x, t_end, mu, c)
        e1_forw_fe.append(E1(reference_fe, ux_forw_fe, x))
        e2_forw_fe.append(E2(reference_fe, ux_forw_fe, x))
        e1_back_fe.append(E1(reference_fe, ux_back_fe, x))
        e2_back_fe.append(E2(reference_fe, ux_back_fe, x))

        # Plot the distribution of the different methods.
        plt.plot(x_fd, u_forw_fd, ':', label=f"FD forward N={N}")
        plt.plot(x_fd, u_back_fd, ':', label=f"FD backward N={N}")
        plt.plot(x, ux_forw_fe, ':', label=f"FE forward N={N}")
        plt.plot(x, ux_back_fe, ':', label=f"FE backward N={N}")
        plt.legend(loc=1)


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
    plt.show()
