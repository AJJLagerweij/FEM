"""
The main FEM loop.

This main FEM loop will assample the differenc matrices that are required in a FEM solver.
It takes the following steps:

1. Loop over all elements.
2. Compute the element based integrals
    1. calculate the quantities in the reference element.
    2. integrate using Quadrature rules and include the mapping from reference to global axis system.
3. Assamble these element based contributions into a global operator.

Bram Lagerweij
COHMAS Mechanical Engineering KAUST
2021
"""

# Import external modules.
import numpy as np
import scipy.sparse as ssp
from numba import njit


def shape1d(x, order=1):
    """
    Shape function for a 1D Lagrange reference element.

    Parameters
    ----------
    x : array_like (float)
        Location where the shape functions are sampled.
    order : int, optional
        The order of the polynomial used. Optional, default value is 1.

    Returns
    -------
    array_like
        The array with the values of the shape funcions at `x`.

    Raises
    ------
    NotImplementedError
        Raised when the requested order of shape function polynomaial is not implemented.
    """
    if order == 1:
        phi1 = 1 - x
        phi2 = x
        list = np.array([phi1, phi2])
    elif order == 2:
        phi1 = 1 - 3 * x + 2 * x ** 2
        phi2 = -x + 2 * x ** 2
        phi3 = 4 * x - 4 * x ** 2
        list = np.array([phi1, phi2, phi3])
    else:
        raise NotImplementedError("This order of shape function is not implemented.")

    return list


def gaus(num):
    """
    Gausian integration points and weights for `num` sample points.

    Computes the sample points and weights for Gauss-Legendre quadrature.
    These sample points and weights will correctly integrate polynomials of
    degree :math:`2*deg - 1` or less over the interval :math:`[0, 1]` with
    the weight function :math:`f(x) = 1`.

    Parameters
    ----------
    num : int
        Number of sample points and weights. It must be >= 1.

    Returns
    -------
    x : array_like
        1D array containing the sample points.
    w : array_like
        1D array containing the weights at the sample points.
    """
    x, w = np.polynomial.legendre.leggauss(num)
    x = (x + 1)/2  # correct locations for intervals [0, 1]
    w = w/2  # correct weight for interval [0, 1]
    return x, w


def get_element(num_q, x, rhs, order=1):
    """
    Get the properties of this element.

    Parameters
    ----------
    num_q : int
        Number of Gausian quadrature points.
    x : array_like(float)
        Nodal locations in global coordinates.
    rhs : callable
        Function that acts as our right hand side (nonhomogeneous term).
    order : int
        Order of the polynomial used by our element.

    Returns
    -------
    wq : array_like(float), shape(nun_q)
        The weighing function for Gausian quadrature.
    phi_xq : array_like(float), shape((dofs, num_q))
        For each shape function the weigts at the quadrature points.
    phi_xq_detJ : array_like(float), shape((dofs, num_q))
        For each shape function times local determinant at the quadrature points.
    f_xq : array_like(float), shape(num_q)
        The value of the right hand side equation evaluated at the quadrature points.
    """
    # Get information on element coordinates.
    h = x[-1] - x[0]
    detJ = h

    # Get the Quadrature weights.
    xq, wq = gaus(num_q)

    # Get properties at quadrature points.
    phi_xq = shape1d(xq, order=order)
    phi_xq_detJ = phi_xq * detJ
    f_xq = rhs(xq*h + x[0])
    return wq, phi_xq, phi_xq_detJ, f_xq


def element_loop(wq, phi_xq, phi_xq_detJ, f_xq):
    """
    Compute the elmement Matrices and Vectors.

    Parameters
    ----------
    wq : array_like(float), shape(nun_q)
        The weighing function for Gausian quadrature.
    phi_xq : array_like(float), shape((dofs, num_q))
        For each shape function the weigts at the quadrature points.
    phi_xq_detJ : array_like(float), shape((dofs, num_q))
        For each shape function times local determinant at the quadrature points.
    f_xq : array_like(float), shape(num_q)
        The value of the right hand side equation evaluated at the quadrature points.

    Returns
    -------
    Me : array_like(float), shape((dofs, dofs))
        Element mass matrix.
    fe : array_like(float), shape(dofs)
        Element right hand side in our system of equations.
    """
    # Create empty storage for element properties.
    dofs = len(phi_xq)
    Me = np.zeros((dofs, dofs))
    fe = np.zeros(dofs)

    # Quadrature summations handeled by numpy.sum().
    # Loop over all degrees of freedom and get:
    for i in range(dofs):
        # Vector quantity.
        fe[i] = np.sum(wq * f_xq * phi_xq_detJ[i])

        # Matrix diagonal quantiy.
        Me[i, i] = np.sum(wq * phi_xq[i] * phi_xq_detJ[i])

        # Loop over all degrees of freedom and get matrix quantities.
        for j in range(i+1, dofs):
            ME_ij = np.sum(wq * phi_xq[j] * phi_xq_detJ[i])
            Me[i, j] = ME_ij
            Me[j, i] = ME_ij
    return Me, fe


def fem1d(x, c, rhs, num_q, order=1):
    """
    Create the global FEM system by looping over the elements.

    Parameters
    ----------
    x : array_like(float)
        Global coordinates of all degrees of freedom.
    c : array_like(int)
        Element nodal coondinate connectivety map.
    rhs : callable
        Function that acts as our right hand side (nonhomogeneous term).
    num_q : int
        Number of Gausian quadrature points.
    order : int
        Order of the polynomial used by our element.

    Returns
    -------
    M : array_like(float), shape((dofs, dofs))
        Global mass matrix.
    f : array_like(float), shape(dofs)
        Global right hand side in our system of equations.
    """
    num_ele = len(c)
    num_dofs = np.max(c) + 1

    # Initialize variables
    f = np.zeros(num_dofs)
    M = np.zeros((num_dofs, num_dofs))

    for ele in range(num_ele):
        # Obtain element properties
        x_ele = x[c[ele]]
        wq, phi_xq, phi_xq_detJ, f_xq = get_element(num_q, x_ele, rhs, order=order)

        # Performe integration and compute element linear algabra objects.
        Me, fe = element_loop(wq, phi_xq, phi_xq_detJ, f_xq)

        # Place these objects into their global counter parts.
        dofs = c[ele]
        f[dofs] += fe
        M[np.ix_(dofs, dofs)] += Me
    return M, f
