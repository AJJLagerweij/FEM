r"""
Approximating

.. math::
    f(x) = \sin^4(2\pi x) \qquad \forall \, x \in \Omega = [0, 1]

With the following approximations functions :math:`f_h(x)`:

1. a Taylor series expansion around :math:`x=0.5`,
2. a Fourier series,
3. a global polynomial and,

Test these approximations using two errors:

.. math::
    E_1 := \int_\Omega | f(x) - f_h(x) | dx \\
    E_2 := \int_\Omega \big(f(x) - f_h(x)\big)^2 dx


Bram Lagerweij
COHMAS Mechanical Engineering KAUST
2021
"""

# Importing External modules? Read our tutorial
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy.fft import rfft, irfft
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


def Taylor_series(N, x):
    r"""
    Returns a polynomial as a Taylor approximation of:

    .. math::
         f(x) = \sin^4(2\pi x)

    around :math:`x=0.5`. The polynomial coefficients are hard coded.

    Parameters
    ----------
    N : int
        Order of the Taylor approximation.
    x : array_like
        Locations where the Taylor series is evaluated, :math:`f_h(x)'.

    Returns
    -------
    array_like
        Taylor_series at points :math:`x`, that is :math:`f_h(x)`.
    """
    # Hardcode coefficients.
    coef = [0, 0, 0, 0, 1558.55, 0, -41019.3, 0, 485813, 0, -3.54021e6, 0,
            1.65587e7, 0, -5.75114e-7, 0, 1.51392e8, 0, -3.12521e8, 0, 5.19494e8]
    coef = coef[:N + 1]  # keep only the coefficients that we want.
    poly = np.poly1d(coef[::-1])

    # Evaluation at points x.
    fh = poly(x - 0.5)
    return fh


def Fourier_coefs(fun, N, x):
    r"""Calculates the first 2*N+1 Fourier series coeff. of a periodic function.

    Given a periodic, function f(x) with period T, this function computes the
    coefficients :math:`a0, {a_1,a_2,...}, {b_1,b_2,...}` such that:

    .. math::
        f_h(x) = \frac{a_0}{2}+ \sum_{k=1}^{N} \big( a_k\cos(2\pi k \frac{t}{T})
                                                 + b_k\sin(2\pi k \frac{t}{T}) \big)

    At the points :math:`x` we evaluate this expression.

    Parameters
    ----------
    fun : callable
        The periodic function, a callable like :math:`f(t)`.
    N : int
        The function will return the first `N + 1` Fourier coeff.
    x : array_like
        The array of points where the fourier series will be evaluated.

    Returns
    -------
    array_like
        Magnitude of the Fourier series approximation :math:`f_h(x)`.
    """
    # From Shanon Theorem we must use a sampling freq. larger than the maximum
    T = 1
    f_sample = 2 * N
    points = np.linspace(0, T, f_sample + 2, endpoint=False)

    # Perform fast Fourier Analysis.
    coef = rfft(fun(points))[:N] / len(points)

    # Assume that x is a regular grid, and that we will ignore the endpoint.
    fh = irfft(coef, len(x) - 1) * (len(x) - 1)
    fh = np.append(fh, [fh[0]])  # add endpoint
    return fh


def Polynomial(fun, N, x):
    r"""
    Compute the best fit polynomial:

    .. math::
        f_h(x) = a_0 + a_1 x + a_2 x^2 + \dots + a_{N-1}x^{N-1}

    which is fitted through :math:`N` evently distributed points.

    Parameters
    ----------
    fun : callable
        The periodic function, a callable like :math:`f(t)`.
    N : int
        The function will evaluate a polynomial of order :math:`N`.
    x : array_like
        The array of points where the fourier series will be evaluated.

    Returns
    -------
    array_like
        Magnitude of the Polynomial fit :math:`f_h(x)`.
    """

    # Obtain locations where we fit the polynomial and fit.
    points = np.linspace(x[0], x[-1], N)
    poly = polyfit(points, fun(points), N - 1)

    # Obtain the result at locations x.
    fh = np.polyval(poly[::-1], x)
    return fh


def Polynomial5(fun, N, x):
    r"""
    Compute the best fit polynomial:

.. math::
        f_h(x) = a_0 + a_1 x + a_2 x^2 + \dots + a_{N-1}x^{N-1}

    which is fitted through :math:`5N` equally distributed points.

    Parameters
    ----------
    fun : callable
        The periodic function, a callable like :math:`f(t)`.
    N : int
        The function will evaluate a polynomial of order :math:`N`.
    x : array_like
        The array of points where the fourier series will be evaluated.

    Returns
    -------
    array_like
        Magnitude of the Polynomial fit :math:`f_h(x)`.
    """

    # Obtain locations where we fit the polynomial and fit.
    points = np.linspace(x[0], x[-1], 5 * N)
    poly = polyfit(points, fun(points), N - 1)

    # Obtain the result at locations x.
    fh = np.polyval(poly[::-1], x)
    return fh


def exact(x):
    fun = np.sin(2 * np.pi * x) ** 4
    return fun


# Compute exact function.
len_x = 10000
x = np.linspace(0, 1, len_x)

plt.figure()
plt.plot(x, exact(x), label='exact', lw=2, zorder=10)

# Store error results.
N_list = []
e1_taylor = []
e2_taylor = []
e1_fourier = []
e2_fourier = []
e1_polynomial = []
e2_polynomial = []
e1_polynomial5 = []
e2_polynomial5 = []

for N in range(1, 21):
    # Range of unknowns.
    N_list.append(N)

    # Taylor Series
    taylor = Taylor_series(N, x)
    e1_taylor.append(E1(exact(x), taylor, x))
    e2_taylor.append(E2(exact(x), taylor, x))
    # plt.plot(x, taylor, label='Taylor '+str(N))

    # Fourier Series
    fourier = Fourier_coefs(exact, N, x)
    e1_fourier.append(E1(exact(x), fourier, x))
    e2_fourier.append(E2(exact(x), fourier, x))
    # plt.plot(x, fourier, label='Fourier '+str(N))

    # Polynomial fit with Least Square Root
    polynomial = Polynomial(exact, N, x)
    e1_polynomial.append(E1(exact(x), polynomial, x))
    e2_polynomial.append(E2(exact(x), polynomial, x))
    # plt.plot(x, polynomial, label='Polynomial '+str(N))

    # Polynomial fit with more datapoints
    polynomial5 = Polynomial5(exact, N, x)
    e1_polynomial5.append(E1(exact(x), polynomial5, x))
    e2_polynomial5.append(E2(exact(x), polynomial5, x))
    # plt.plot(x, polynomial, label='Polynomial '+str(N))

plt.ylim(-0.5, 1.5)
plt.ylabel('$f(x)$ and $f_h(x)$')
plt.xlim(0, 1)
plt.xlabel('$x$')
plt.legend(loc=1)

# Plotting the errors.
plt.figure()
plt.plot(N_list, e1_taylor, 's', label='Taylor')
plt.plot(N_list, e1_fourier, 'o', label='Fourier')
plt.plot(N_list, e1_polynomial, '^', label='Polynomial')
plt.plot(N_list, e1_polynomial5, 'v', label='Polynomial 5')
plt.yscale('log')
plt.ylabel('$E_1$')
plt.xticks(np.arange(0, N + 1, 1))
plt.xlim(0, N + 1)
plt.xlabel('N')
plt.legend(loc=3)

plt.figure()
plt.plot(N_list, e2_taylor, 's', label='Taylor')
plt.plot(N_list, e2_fourier, 'o', label='Fourier')
plt.plot(N_list, e2_polynomial, '^', label='Polynomial')
plt.plot(N_list, e2_polynomial5, 'v', label='Polynomial 5')
plt.yscale('log')
plt.ylabel('$E_2$')
plt.xticks(np.arange(0, N + 1, 1))
plt.xlim(0, N + 1)
plt.xlabel('N')
plt.legend(loc=3)
plt.show()
