r"""
Approximating

.. math::
    f(x) = \begin{cases} 1 & 0.35 \leq x \leq 0.65 \\ 0 & \text{otherwise} \end{cases}

Onto a FE space, the approximation :math:`f_h(x)` defined in finite sized elements using:

1. continuous linear polynomials
2. continuous quadratic polynomials

Test these approximations using two errors:

.. math::
    E_1 := \int_\Omega | f(x) - f_h(x) | dx \\
    E_2 := \sqrt(\int_\Omega \big(f(x) - f_h(x)\big)^2) dx

Bram Lagerweij
COHMAS Mechanical Engineering KAUST
2021
"""

# Importing modules.
import sys

# Importing modules.
import sys
import matplotlib.pyplot as plt
import numpy as np
import numba as nb
nb.NUMBA_DISABLE_JIT = 0  # Turn JIT on (0) and off (1)

# Importing my own modules
sys.path.insert(1, '../src')
from pde import projection
from solvers import solve
from element import Mesh1D
from fem import interpolate
from helper import E1, E2


@nb.jit(nopython=True)
def exact(x):
    r"""
    The exact function:

    .. math::
        f(x) = \begin{cases} 1 & 0.35 \leq x \leq 0.65 \\ 0 & \text{otherwise} \end{cases}

    Parameters
    ----------
    x : array_like(float)
        Locations :math:`x` where the function is evaluated.

    Returns
    -------
    array_like(float)
        The function :math:`f(x)` at points `x`.
    """
    fun = np.zeros_like(x)
    index = np.where((0.35<x) & (x<0.65))
    fun[index] = 1
    return fun


if __name__ == '__main__':
    # Setup grid to measure exact results and errors.
    len_x = int(1e6)
    x = np.linspace(0.0, 1.0, len_x)

    # Store error results.
    N_list = 2**np.arange(2, 10)
    e1_linear = []
    e2_linear = []
    e1_quadratic = []
    e2_quadratic = []
    num_q = 4

    # Compute and compare Linear elements.
    lin_plot = plt.figure(num='Linear Elements')
    lin_ax = plt.gca()
    lin_ax.plot(x, exact(x), lw=2, label='Exact')

    print("Linear")
    for N in N_list:
        print(f'{N} Elements')
        mesh_lin = Mesh1D(0.0, 1.0, N, 1, num_q)  # Setup
        pde = projection(mesh_lin, exact)  # Assemble
        u = solve(*pde)  # Solve

        u_x = interpolate(mesh_lin, u, x)  # Post process
        lin_ax.plot(x, u_x, ':', label=f'{N} elements')
        e1_linear.append(E1(exact(x), u_x, x))
        e2_linear.append(E2(exact(x), u_x, x))

    lin_ax.set_ylim(0, 1.1)
    lin_ax.set_ylabel('$f(x)$ and $f_h(x)$')
    lin_ax.set_xlim(0, 1)
    lin_ax.set_xlabel('$x$')
    lin_ax.legend(loc=1)
    lin_plot.tight_layout()

    # Compute and compare Quadratic elements.
    qua_plot = plt.figure(num='Quadratic Elements')
    qua_ax = plt.gca()
    qua_ax.plot(x, exact(x), lw=2, label='Exact')

    print("Quadratic")
    for N in N_list:
        print(f'{N} Elements')
        mesh_qua = Mesh1D(0.0, 1.0, N, 2, num_q)  # Setup
        pde = projection(mesh_qua, exact)  # Assemble
        u = solve(*pde)  # Solve

        u_x = interpolate(mesh_qua, u, x)  # Post process
        qua_ax.plot(x, u_x, ':', label=f'{N} elements')
        e1_quadratic.append(E1(exact(x), u_x, x))
        e2_quadratic.append(E2(exact(x), u_x, x))

    qua_ax.set_ylim(0, 1.1)
    qua_ax.set_ylabel('$f(x)$ and $f_h(x)$')
    qua_ax.set_xlim(0, 1)
    qua_ax.set_xlabel('$x$')
    qua_ax.legend(loc=1)
    qua_plot.tight_layout()

    # Plotting the errors vs number of elements.
    plt.figure(num='E1 vs Elements')
    plt.plot(N_list, e1_linear, 's', label='Linear')
    plt.plot(N_list, e2_quadratic, 'o', label='Quadratic')
    plt.plot(N_list, 1/N_list**1, ':', label='$N^{-1}$')
    plt.plot(N_list, 1/N_list**2, ':', label='$N^{-2}$')
    plt.plot(N_list, 1/N_list**3, ':', label='$N^{-3}$')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('$E_1$')
    plt.xlabel('Number of Elements')
    plt.legend(loc=3)
    plt.tight_layout()

    plt.figure(num='E2 vs Elements')
    plt.plot(N_list, e2_linear, 's', label='Linear')
    plt.plot(N_list, e2_quadratic, 'o', label='Quadratic')
    plt.plot(N_list, 1/N_list**1, ':', label='$N^{-1}$')
    plt.plot(N_list, 1/N_list**2, ':', label='$N^{-2}$')
    plt.plot(N_list, 1/N_list**3, ':', label='$N^{-3}$')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('$E_2$')
    plt.xlabel('Number of Elements')
    plt.legend(loc=3)
    plt.tight_layout()

    # Plotting the errors vs degrees of freedom.
    plt.figure(num='E1 vs DOFs')
    plt.plot(N_list + 1, e1_linear, 's', label='Linear')
    plt.plot(2*N_list + 1, e2_quadratic, 'o', label='Quadratic')
    plt.plot(N_list, 1/N_list**1, ':', label='$dof^{-1}$')
    plt.plot(N_list, 1/N_list**2, ':', label='dof$^{-2}$')
    plt.plot(N_list, 1/N_list**3, ':', label='dof$^{-3}$')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('$E_1$')
    plt.xlabel('Degrees of Freedom')
    plt.legend(loc=3)
    plt.tight_layout()

    plt.figure(num='E2 vs DOFs')
    plt.plot(N_list, e2_linear, 's', label='Linear')
    plt.plot(2*N_list, e2_quadratic, 'o', label='Quadratic')
    plt.plot(N_list, 1/N_list**1, ':', label='$dof^{-1}$')
    plt.plot(N_list, 1/N_list**2, ':', label='dof$^{-2}$')
    plt.plot(N_list, 1/N_list**3, ':', label='dof$^{-3}$')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('$E_2$')
    plt.xlabel('Degrees of Freedom')
    plt.legend(loc=3)
    plt.tight_layout()
    plt.show()
