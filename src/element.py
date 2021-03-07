r"""
Shape functions for in reference coordinate system.

Bram Lagerweij
COHMAS Mechanical Engineering KAUST
2021
"""

# Importing required modules.
import numba as nb
import numpy as np

# Import my own scripts.
from helper import gauss


@nb.jit(nopython=True)
def shape1d(x, order):
    r"""
    Shape function for a 1D Lagrange reference element.

    Parameters
    ----------
    x : array_like (float)
        Location where the shape functions are sampled.
    order : int
        The order of the polynomial used.

    Returns
    -------
    array_like
        The array with the values of the shape funcions at `x`.

    Raises
    ------
    NotImplementedError
        Raised when the requested order of shape function polynomaial is not implemented.
    """
    phi = np.zeros((order+1, len(x)))
    dphi = np.zeros_like(phi)

    if order == 1:
        # Shape functions
        phi[0] = 1 - x
        phi[1] = x

        # Derivatives
        dphi[0] = -1
        dphi[1] = 1

    elif order == 2:
        # Shape functions
        phi[0] = 1 - 3 * x + 2 * x ** 2
        phi[2] = -x + 2 * x ** 2
        phi[1] = 4 * x - 4 * x ** 2

        # Derivatives
        dphi[0] = -3 + 4 * x
        dphi[2] = -1 + 4 * x
        dphi[1] = 4 - 8 * x

    else:
        raise NotImplementedError("This order of shape function is not implemented.")

    return phi, dphi


def triangle2d(x, order):
    r"""
    Shape function for a 2D linear triangles.

    Parameters
    ----------
    x : array_like (float, float)
        Location [x1, x2] where the shape functions are sampled.
    order : int
        The order of the polynomial used, only option is 1.

    Returns
    -------
    array_like
        The array with the values of the shape funcions at `x`.

    Raises
    ------
    NotImplementedError
        Raised when the requested order of shape function polynomaial is not implemented.
    """
    phi = np.zeros((order+1, len(x)))
    dxphi = np.zeros_like(phi)
    dyphi = np.zeros_like(phi)

    if order == 1:
        # Shape functions
        phi[0] = x[0]
        phi[1] = x[1]
        phi[2] = 1 - x[0] - x[1]

        # Derivatives
        dxphi[0] = 1
        dxphi[1] = 0
        dxphi[2] = -1
        dyhpi[0] = 0
        dyhpi[0] = 1
        dyhpi[0] = -1

    else:
        raise NotImplementedError("This order of shape function is not implemented.")

    return phi, dphi


@nb.jit(nopython=False)
def interpolate(u, x, c, x_inter, order):
    r"""
    Obtain the field :math`u(x)` any points `x_inter` following the FE interpolation.

    Parameters
    ----------
    u : array_like(float), shape(dofs)
        The field `u` at the degrees of freedom.
    x : array_like(float), shape(dofs)
        The location of the degrees of freedom.
    c : array_like(int), shape((num_ele, dofe/ele))
        Element to degree of freedom connectivety map.
    x_inter : array_like(float)
        The location where we want to obtain our interpolated field.
    order : int
        The order of the shape functions used.

    Returns
    -------
    array_like(float)
        The field `u` at the interpolation points `x_inter`.
    """
    # Obtain knowledge about the size of the problems.
    num_ele = len(c)

    # Initialize storage for our interpolated field.
    u_inter = np.zeros_like(x_inter)

    # Loop over all elements and interpolate the inside.
    for ele in range(num_ele):
        # Obtain element properties
        dofe = c[ele]
        x_ele = x[dofe]

        # Find which values of x are within our element.
        ind = np.where((x_ele.min() <= x_inter) & (x_inter <= x_ele.max()))
        x_loc = (x_inter[ind] - x_ele[0]) / (x_ele[-1] - x_ele[0])

        # Obtain the shape functions and calculate the field.
        phi_x, dphi_x = shape1d(x_loc, order)
        u_inter[ind] = np.sum(u[dofe] * phi_x.T, axis=1)

    return u_inter


@nb.jit(nopython=True)
def get_element(num_q, x, rhs, order):
    r"""
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
    phi_xq : array_like(float), shape((dofs, num_q))
        For each shape function the value at the quadrature points.
    invJ_dphi_xq : array_like(float), shape((dofs, num_q))
        For each shape function its derivative value at the quadrature points times the inverse Jacobian.
    f_xq : array_like(float), shape(num_q)
        The value of the right hand side equation evaluated at the quadrature points.
    wq_detJ : array_like(float), shape((dofs, num_q))
        For the local determinant times quadrature weight at each of the quadrature points.
    """
    # Get information on element coordinates.
    h = x[-1] - x[0]
    detJ = h
    invJ = 1/h

    # Get the Quadrature weights.
    xq, wq = gauss(num_q)

    # Get properties at quadrature points.
    phi_xq, dphi_xq = shape1d(xq, order)
    invJ_dphi_xq = invJ * dphi_xq
    wq_detJ = wq * detJ

    # Check if the right hand side is a function or not.
    if rhs != None:
        f_xq = rhs(xq*h + x[0])
    else:
        f_xq = np.zeros_like(xq)
    return phi_xq, invJ_dphi_xq, f_xq, wq_detJ
