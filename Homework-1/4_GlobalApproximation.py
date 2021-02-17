r"""
Approximating

.. math::
    f(x) = \sin^4(2\pi x) \qquad \forall \, x \in \Omega = [0, 1]

With the following local approximation functions :math:`f_h(x)` defined in finite sized cells:
    1. using linear curve inside the cells and,
    2. using quadratic curves inside the cells.

Test these approximations using two errors:

.. math::
    E_1 := \int_\Omega | f(x) - f_h(x) | dx \\
    E_2 := \int_\Omega \big(f(x) - f_h(x)\big)^2 dx


Bram Lagerweij
COHMAS Mechanical Engineering KAUST
2021
"""

# Importing modules.
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson


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
        E_2 := \int_\Omega \big(f(x) - f_h(x)\big)^2 dx

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
    integral = simpson(e, x)
    return integral


def linear_hat(x, start, center, end):
    """
    The linear hat function of a node that is located at center, start/end the previous and next nodes.

    Parameters
    ----------
    x : array_like
        Location where we evaluate the hat functions.
    start : float
        Location of the left neighbouring node, till here the function will have zero weights.
    center : float
        Location of the node at this location the weights will be one.
    end : float
        Location of the right neighbouring dode, from here on the weights will be zero.

    Returns
    -------
    phi : array_like
        The shape function relate to this node.
    """
    # Coordinates that are part of our node
    phi = np.zeros_like(x)
    index1 = np.where((start <= x) & (x <= center))
    index2 = np.where((center <= x) & (x <= end))

    # Create the shape function
    if start != center:
        phi[index1] = (x[index1] - start) / (center - start)
    if center != end:
        phi[index2] = 1 - (x[index2] - center) / (end - center)

    return phi


def Linear(fun, x, N):
    """
    Linear element approximation.

    Parameters
    ----------
    fun : callable
        The function that we are approximation.
    x : array_like
        Locations where we want to approximate the function.
    N : int
        Number of elements.

    Returns
    -------
    fh : array_like
        The approximation :math:`f_h(x)`.
    """
    fh = np.zeros_like(x)

    grid = np.linspace(0, 1, N+1)
    for n in range(N + 1):
        node_center = grid[n]

        if n == 0:  # If the shape function consist of only the second half
            node_start = grid[n]
        else:
            node_start = grid[n - 1]

        if n == N:  # If the shape function consists of only the first half.
            node_end = grid[n]
        else:
            node_end = grid[n + 1]

        fh += fun(node_center) * linear_hat(x, node_start, node_center, node_end)
    return fh


def quadratic_hat(x, start, center, end):
    """
    The quadratic hat function of a node that is located at center, start/end the previous and next nodes.

    Parameters
    ----------
    x : array_like
        Location where we evaluate the hat functions.
    start : float
        Location of the left neighbouring node, till here the function will have zero weights.
    center : float
        Location of the node at this location the weights will be one.
    end : float
        Location of the right neighbouring dode, from here on the weights will be zero.

    Returns
    -------
    phi : array_like
        The shape function relate to this node.
    """
    # Coordinates that are part of our node
    phi = np.zeros_like(x)
    index1 = np.where((start <= x) & (x <= center))
    index2 = np.where((center <= x) & (x <= end))

    # Create the shape function
    if start != center:
        middle_left = start + (center - start) / 2
        phi[index1] = 2*(x[index1] - start) * (x[index1] - middle_left) / (center - start)**2
    if center != end:
        middle_right = end - (end - center) / 2
        phi[index2] = 2*(x[index2] - end) * (x[index2] - middle_right) / (end - center)**2
    return phi


def quadratic_hat_center(x, start, end):
    """
    The quadratic hat function of a minor node in the middle of an element.

    Parameters
    ----------
    x : array_like
        Location where we evaluate the hat functions.
    start : float
        Location of the left neighbouring node, till here the function will have zero weights.
    end : float
        Location of the right neighbouring dode, from here on the weights will be zero.

    Returns
    -------
    phi : array_like
        The shape function relate to this node.
    """
    # Coordinates that are part of our node
    phi = np.zeros_like(x)
    index = np.where((start <= x) & (x <= end))

    # Create the shape function
    phi[index] = -4*(x[index] - start) * (x[index] - end) / (end-start)**2
    return phi


def Quadratic(fun, x, N):
    """
    Quadratic element approximation.

    Parameters
    ----------
    fun : callable
        The function that we are approximation.
    x : array_like
        Locations where we want to approximate the function.
    N : int
        Number of elements.

    Returns
    -------
    fh : array_like
        The approximation :math:`f_h(x)`.
    """
    fh = np.zeros_like(x)

    grid = np.linspace(0, 1, N+1)
    for n in range(N + 1):
        node_center = grid[n]

        if n == 0:  # If the shape function consist of only the second half
            node_start = grid[n]
        else:
            node_start = grid[n - 1]

        if n == N:  # If the shape function consists of only the first half.
            node_end = grid[n]
        else:
            node_end = grid[n + 1]

        fh += exact(node_center) * quadratic_hat(x, node_start, node_center, node_end)

    # Middle shape functions of an element, these are of the interior node.
    for n in range(N):
        node_start = grid[n]
        node_end = grid[n + 1]
        node_center = node_start + (node_end - node_start) / 2
        fh += exact(node_center) * quadratic_hat_center(x, node_start, node_end)
    return fh


def exact(x):
    fun = np.sin(2 * np.pi * x) ** 4
    return fun


# Compute exact function.
len_x = 10001
x = np.linspace(0, 1, len_x)

plt.figure()
plt.plot(x, exact(x), lw=2, zorder=10, label='Exact')

# Store error results.
N_list = []
e1_linear = []
e2_linear = []
e1_quadratic = []
e2_quadratic = []

for N in range(1, 21):
    N_list.append(N)

    # Linear fits.
    linear = Linear(exact, x, N)
    e1_linear.append(E1(exact(x), linear, x))
    e2_linear.append(E2(exact(x), linear, x))
    # plt.plot(x, linear, label=f'Linear {N}')

    # Quadratic fit
    quadratic = Quadratic(exact, x, N)
    e1_quadratic.append(E1(exact(x), linear, x))
    e2_quadratic.append(E2(exact(x), linear, x))
    # plt.plot(x, quadratic, label=f'Quadratic {N}')


plt.ylim(-0.25, 1.25)
plt.ylabel('$f(x)$ and $f_h(x)$')
plt.xlim(0, 1)
plt.xlabel('$x$')
plt.legend(loc=1)

# Plotting the errors.
plt.figure()
plt.plot(N_list, e1_linear, 's', label='Linear')
plt.plot(N_list, e2_quadratic, 'o', label='Quadratic')
plt.yscale('log')
plt.ylabel('$E_1$')
plt.xticks(np.arange(0, N + 1, 1))
plt.xlim(0, N + 1)
plt.xlabel('N')
plt.legend(loc=3)

plt.figure()
plt.plot(N_list, e2_linear, 's', label='Linear')
plt.plot(N_list, e1_quadratic, 'o', label='Quadratic')
plt.yscale('log')
plt.ylabel('$E_2$')
plt.xticks(np.arange(0, N + 1, 1))
plt.xlim(0, N + 1)
plt.xlabel('N')
plt.legend(loc=3)
plt.show()