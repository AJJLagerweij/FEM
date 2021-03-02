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
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson
from scipy.linalg import solve

# Importing my own modules
sys.path.insert(1, '../src')
from fem import fem1d


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


def mesh(N):
    nodes = np.linspace(0, 1, N+1)
    ele, dof = np.indices((N, 2))
    connectivity = ele + dof
    return nodes, connectivity


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

N = 128
num_q = 4

# Create mesh
grid, connect = mesh(N)

# Obtain global system of equations
M, f = fem1d(grid, connect, exact, num_q, order=1)

# Solve linear system of equations
u = solve(M, f)


# Plot result for comparison to exact solution.
x = np.linspace(0, 1, 1001)
fig = plt.figure()
plt.plot(x, exact(x), label='exact')
plt.plot(grid, u, ':o')
plt.title(f'{N} elements')
plt.show()



# # Compute exact function.
# len_x = 11
# x = np.linspace(0, 1, len_x)

# plt.figure()
# plt.plot(x, exact(x), lw=2, zorder=10, label='Exact')

# # Store error results.
# N_list = []
# e1_linear = []
# e2_linear = []
# e1_quadratic = []
# e2_quadratic = []

# for N in range(1, 21):
#     N_list.append(N)
#
#     # Linear fits.
#     linear = Linear(exact, x, N)
#     e1_linear.append(E1(exact(x), linear, x))
#     e2_linear.append(E2(exact(x), linear, x))
#     plt.plot(x, linear, label=f'Linear {N}')
#
#     # Quadratic fit
#     quadratic = Quadratic(exact, x, N)
#     e1_quadratic.append(E1(exact(x), quadratic, x))
#     e2_quadratic.append(E2(exact(x), quadratic, x))
#     plt.plot(x, quadratic, label=f'Quadratic {N}')
#
#
# plt.ylim(-0.25, 1.25)
# plt.ylabel('$f(x)$ and $f_h(x)$')
# plt.xlim(0, 1)
# plt.xlabel('$x$')
# plt.legend(loc=1)
#
# # Plotting the errors.
# plt.figure()
# plt.plot(N_list, e1_linear, 's', label='Linear')
# plt.plot(N_list, e2_quadratic, 'o', label='Quadratic')
# plt.yscale('log')
# plt.ylabel('$E_1$')
# plt.xticks(np.arange(0, N + 1, 1))
# plt.xlim(0, N + 1)
# plt.xlabel('N')
# plt.legend(loc=3)
#
# plt.figure()
# plt.plot(N_list, e2_linear, 's', label='Linear')
# plt.plot(N_list, e2_quadratic, 'o', label='Quadratic')
# plt.yscale('log')
# plt.ylabel('$E_2$')
# plt.xticks(np.arange(0, N + 1, 1))
# plt.xlim(0, N + 1)
# plt.xlabel('N')
# plt.legend(loc=3)
# plt.show()
