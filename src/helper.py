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
def gaussquad(num):
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
    xi : array_like(float)
        1D array containing the sample points.
    w : array_like(float)
        1D array containing the weights at the sample points.
    """
    if num == 2:
        xi = np.array([-0.5773502691896257645092, 0.5773502691896257645092])
        w = np.array([1., 1.])
    if num == 3:
        xi = np.array([-0.7745966692414833770359, 0, 0.7745966692414833770359])
        w = np.array([0.5555555555555555555556, 0.8888888888888888888889, 0.555555555555555555556])
    elif num == 4:
        xi = np.array([-0.861136311594052575224, -0.3399810435848562648027,
                      0.3399810435848562648027, 0.861136311594052575224])
        w = np.array([0.3478548451374538573731, 0.6521451548625461426269,
                      0.6521451548625461426269, 0.3478548451374538573731])
    elif num == 5:
        xi = np.array([-0.9061798459386639927976, -0.5384693101056830910363, 0,
                      0.5384693101056830910363, 0.9061798459386639927976])
        w = np.array([0.2369268850561890875143, 0.4786286704993664680413, 0.5688888888888888888889,
                      0.4786286704993664680413, 0.2369268850561890875143])
    else:
         raise NotImplementedError("This number of quadrature points is not implemented.")

    # Correct weights and coordinates for interval [0, 1]
    xi = (xi + 1) / 2
    w = w / 2
    return xi, w


def quadtri(num):
    """
    Symetric quadrature points and weights for `num` sample points in a triangle.

    Computes the sample points and weights.
    These sample points and weights will correctly integrate polynomials of:

    .. table:: : Quadrature with `num` points results in exact integrals for polynomial of order :math:`p`.
        :name: TriangularQuadrature
        :align: center

        +-----------+---+---+---+---+
        | `num`     | 1 | 3 | 4 | 7 |
        +-----------+---+---+---+---+
        | :math:`p` | 1 | 2 | 3 | 4 |
        +-----------+---+---+---+---+

    Parameters
    ----------
    num : int
        Number of sample points and weights. It must be  1, 3, 4, or 7.

    Returns
    -------
    xi : array_like(float)
        1D array containing the sample points in local coordinates.
    w : array_like(float)
        1D array containing the weights at the sample points.
    """
    if num == 1:
        xi = np.array([0.5])
        w = np.array([[1/3, 1/3]])
    if num == 3:
        xi = np.array([1/6, 1/6, 1/6])
        w = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])
    elif num == 4:
        xi = np.array([-9/32, 25/96, 25/96, 25/96])
        w = np.array([[1/3, 1/3], [3/5, 1/5], [1/5, 3/5], [1/5, 1/5]])
    elif num == 7:
        xi = np.array([1/40, 1/15, 1/40, 1/15, 1/40, 1/15, 9/40])
        w = np.array([[0, 0], [1/2, 0], [1, 0], [1/2, 1/2], [0, 1], [0, 1], [0, 1/2], [1/3, 1/3]])
    else:
         raise NotImplementedError("This number of quadrature points is not implemented.")

    return xi, w


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
