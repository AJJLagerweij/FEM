r"""
Approximating

.. math::
    f(x) = \sin^4(2\pi x) \qquad \forall \, x \in \Omega = [0, 1]

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
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# Importing my own modules
sys.path.insert(1, '../src')
from fem import fem1d
from element import interpolate


def E1(fun, fun_h, x):
    r"""
    Calculate the :math:`E_1` error.

    .. math::
        E_1 := \int_\Omega | f(x) - f_h(x) | dx


    Parameters
    ----------
    fun : array_like
        The solution of the exact equation at location :math:`x`.
    fun_h : array_like
        The solution of the approximation equation at location :math:`x`.
    x : array_like
        The locations where the function is analyzed.

    Returns
    -------
    float
        Error of the approximation.
    """
    e = np.abs(fun - fun_h)
    integral = simpson(e, x)
    return integral


def E2(fun, fun_h, x):
    r"""
    Calculate the :math:`E_2` error.

    .. math::
        E_2 := \sqrt(\int_\Omega \big(f(x) - f_h(x)\big)^2 dx)

    Parameters
    ----------
    fun : array_like
        The solution of the exact equation at location :math:`x`.
    fun_h : array_like
        The solution of the approximation equation at location :math:`x`.
    x : array_like
        The locations where the function is analyzed.

    Returns
    -------
    float
        Error of the approximation.
    """
    e = (fun - fun_h) ** 2
    integral = np.sqrt(simpson(e, x))
    return integral


def mesh(x_start, x_end, n, order=1):
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
    order : int, optional
        Order of the interpolation functions.

    Returns
    -------
    nodes : array_like(float)
        Coordinates of all the nodal points.
    connectivity array_like(int), shape((n+1, order+1))
        Elements to node connectivity array.
    """
    ele, dof = np.indices((n, order+1))
    connectivity = order*ele + dof
    ndofs = connectivity.max()+1
    nodes = np.linspace(x_start, x_end, ndofs)
    return nodes, connectivity


@nb.jit(nopython=True)
def exact(x):
    """
    The exact function:

    .. math::
        f(x) = \sin^4(2\pi x)

    Parameters
    ----------
    x : array_like
        Locations :math:`x` where the function is evaluated.

    Returns
    -------
    array_like
        The function :math:`f(x)` at points `x`.
    """
    fun = np.sin(2 * np.pi * x) ** 4
    return fun


# Setup grid to measure exact results and errors.
len_x = int(1e6)
x = np.linspace(0, 1, len_x)

# Store error results.
N_list = 2**np.arange(1, 18)
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
    grid, connect = mesh(0, 1, N, order=1)
    u = fem1d(grid, connect, exact, num_q, order=1, mass=True)
    u_x = interpolate(u, grid, connect, x, order=1)

    lin_ax.plot(x, u_x, ':', label=f'{N} elements')
    e1_linear.append(E1(exact(x), u_x, x))
    e2_linear.append(E2(exact(x), u_x, x))

lin_ax.set_ylim(-0.25, 1.25)
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
    grid, connect = mesh(0, 1, N, order=2)
    u = fem1d(grid, connect, exact, num_q, order=2, mass=True)
    u_x = interpolate(u, grid, connect, x, order=2)

    qua_ax.plot(x, u_x, ':', label=f'{N} elements')
    e1_quadratic.append(E1(exact(x), u_x, x))
    e2_quadratic.append(E2(exact(x), u_x, x))

qua_ax.set_ylim(-0.25, 1.25)
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
plt.show()

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
plt.show()
