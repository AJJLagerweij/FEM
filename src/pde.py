r"""
Storing various PDEs that can be will be solved in this course. This includes:

- Diffusive 1D

    .. math::
       u_{t} = \mu u_{xx} \qquad \forall \, x \in \Omega = [0, L] \quad \& \quad t>0

- Advective 1D

    .. math::
       u_{t} + c {u}_{x} = 0 \qquad \forall \, x \in \Omega = [0, L] \quad \& \quad t>0

- Diffusive-Advective 1D

    .. math::
       u_{t} + c {u}_{x} = \mu u_{xx} \qquad \forall \, x \in \Omega = [0, L] \quad \& \quad t>0

- Poisson in 1D

    .. math::
       u_{uu} = f(x) \qquad \forall \, x \in \Omega = [0, L]

The goal is to implement the code in python and not rely on existing methods.

Bram Lagerweij
COHMAS Mechanical Engineering KAUST
2021
"""

# Importing External modules
import numpy as np
import scipy.sparse as sparse

# Importing my own scripts
from fem import kernel1d


def projection(x, c, fun, num_q, order):
    r"""
    Projecting a 1D function :math:`f(x)` on a finite element basis.

    Lets create our approximation function,

    .. math::
        f_h(x) = \sum_{n=0}^N u_n \phi_n(x)

    as a weighted summation of the basisfunctions of approximation.
    Where :math:`phi_n` are the basisfunctions of our FE space :math:`V_h`.
    The unknows here are the weights :math:`f_n`.
    To find these weights we formulate a weak form:

    .. math::
        \int_\Omega (f_h(x) - f(x)) \, \phi_i(x)\, dV = 0 \quad \forall \quad \phi_i \in V_h

    in which we substitute our approximation function and separate the knowns from the unknowns.
    We find:

    .. math::
         \int_\Omega \phi_i(x) \sum_{j=0}^N u_j \phi_j(x)\, dV = \int_\Omega \phi_i(x) f(x) \, dV \quad \forall \quad \phi_i \in V_h

    As the weights :math:`u_n` are independent of location, we can take them out of the integral:

    .. math::
        \sum_{j=0}^N u_j \int_\Omega \phi_i(x) \phi_j(x)\, dV = \int_\Omega \phi_i(x) f(x)\, dV \quad \forall \quad \phi_i \in V_h

    Which can be rewritten as a system of linear equations, which is:

    .. math::
        M \, u = b

    Where :math:`M` is a matrix and :math:`u` and :math:`b` are vectors.

    Parameters
    ----------
    x : array_like(float)
        Global coordinates of all degrees of freedom.
    c : array_like(int), shape((num_ele, dofe/ele))
        Element to degree of freedom connectivety map.
    fun : callable
        Function that acts as our right hand side (nonhomogeneous term).
    num_q : int
        Number of Gausian quadrature points.
    order : int
        Order of the polynomial used by our element.

    Returns
    -------
    M : matrix, (sparse csr format)
        The mass matrix.
    b : vector, (dense array)
        The right hand side, caused by the non-homogeneous behavior.
    """
    # Obtain matrix elements from main FEM kernel.
    M, T, S, b = kernel1d(x, c, fun, num_q, order, mass=True)

    # Convert matrix objects to sparse counterparts.
    M = sparse.coo_matrix(M).tocsr()
    return M, b


def diffusive(x, c, mu, num_q, order):
    r"""
    Time derivative of the PDE for advective diffusive problems.

    .. math::
        u_{t} = \mu u_{xx}  \qquad \forall \, x \in \Omega = [0, L] \quad \& \quad t>0


    Parameters
    ----------
    x : array_like(float)
        Global coordinates of all degrees of freedom.
    c : array_like(int), shape((num_ele, dofe/ele))
        Element to degree of freedom connectivety map.
    mu : float
        Diffusive constant.
    num_q : int
        Number of Gausian quadrature points.
    order : int
        Order of the polynomial used by our element.

    Returns
    -------
    M : matrix, (sparse csr format)
        The mass matrix.
    S : matrix, (sparse csr format)
        The stiffeness matrix.
    b : vector, (dense array)
        The right hand side, caused by the non-homogeneous behavior.
    """
    # Obtain matrix elements from main FEM kernel.
    M, T, S, b = kernel1d(x, c, fun, num_q, order, mass=True, stiffness=True)

    # Convert matrix objects to sparse counterparts.
    M = sparse.coo_matrix(M).tocsr()
    S = mu * sparse.coo_matrix(S).tocsr()
    return M, S, b
