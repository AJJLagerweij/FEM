r"""
Minor helper functions for FEM problems.

Bram Lagerweij
COHMAS Mechanical Engineering KAUST
2021
"""

# Importing modules
import numba as nb
import numpy as np
from scipy.integrate import simpson


@nb.jit(nopython=True)
def gauss(num):
    """
    Gaussian integration points and weights for `num` sample points.

    Computes the sample points and weights for Gauss-Legendre quadrature.
    These sample points and weights will correctly integrate polynomials of
    degree :math:`2*deg - 1` or less over the interval :math:`[0, 1]` with
    the weight function :math:`f(x) = 1`.

    Parameters
    ----------
    num : int
        Number of sample points and weights. It must be  1 <= num <= 5.

    Returns
    -------
    x : array_like
        1D array containing the sample points.
    w : array_like
        1D array containing the weights at the sample points.
    """
    if num == 2:
        x = np.array([-0.5773502691896257645092, 0.5773502691896257645092])
        w = np.array([1, 1])
    if num == 3:
        x = np.array([-0.7745966692414833770359, 0, 0.7745966692414833770359])
        w = np.array([0.5555555555555555555556, 0.8888888888888888888889, 0.555555555555555555556])
    elif num == 4:
        x = np.array([-0.861136311594052575224, -0.3399810435848562648027,
                      0.3399810435848562648027, 0.861136311594052575224])
        w = np.array([0.3478548451374538573731, 0.6521451548625461426269,
                      0.6521451548625461426269, 0.3478548451374538573731])
    elif num == 5:
        x = np.array([-0.9061798459386639927976, -0.5384693101056830910363, 0,
                      0.5384693101056830910363, 0.9061798459386639927976])
        w = np.array([0.2369268850561890875143, 0.4786286704993664680413, 0.5688888888888888888889,
                      0.4786286704993664680413, 0.2369268850561890875143])
    else:
         raise NotImplementedError("This number of quadrature points is not implemented.")

    # Correct weights and coordinates for interval [0, 1]
    x = (x + 1) / 2
    w = w / 2
    return x, w


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
        E_2 := \sqrt{\int_\Omega \big(f(x) - f_h(x)\big)^2 dx}

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
