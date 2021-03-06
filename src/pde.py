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
from derivative import Dx, Dxx
from fem import kernel1d


def diffusive_fd(dof, dx, mu):
    r"""
    Time derivative of the PDE for advective diffusive problems.

    .. math::
        u_{t} = \mu u_{xx}  \qquad \forall \, x \in \Omega = [0, L] \quad \& \quad t>0

    Thus this returns:

    .. math::
        u_{t} = \mu u_{xx}

    Because we use finite difference based matrix products we can convert this
    into a matrix vector product, where :math:`D_xx` is the central difference
    approximation of :math:`\partial_{xx}`:

    .. math::
        u_{t} = \mu D_{xx} u = K u

    This function calculates the matrix :math:`K`. Because it should be
    compatible with general, non-homogeneous formulation, a part that is
    independent of :math:`u` is also included.

    Parameters
    ----------
    dof : int
        Number of degrees of freedom.
    dx : float
        Step size in the of spatial discretization.
    mu : float
        The diffusive coefficient.

    Returns
    -------
    K : matrix (sparse csr format)
        The time derivative part of the pde obtained from the spatial part.
    b : vector (dense array)
        The remaining term, in this homogeneous case it is a zero array.
    """
    K = mu * Dxx(dof, dx)
    b = np.zeros(dof)
    return K, b


def advective_fd(dof, dx, c):
    r"""
    Time derivative of the PDE for advective diffusive problems.

    .. math::
        u_{t} + c u_{x} = 0  \qquad \forall \, x \in \Omega = [0, L] \quad \& \quad t>0

    Thus this returns:

    .. math::
        u_{t} = - c u_{x}

    Because we use finite difference based matrix products we can convert this
    into a matrix vector product, where :math:`D_x` is the central difference
    approximation of :math:`\partial_x`:

    .. math::
        u_{t} = -c D_{x} u = K u

    This function calculates the matrix :math:`K`. Because it should be
    compatible with general, non-homogeneous formulation, a part that is
    independent of :math:`u` is also included.

    Parameters
    ----------
    dof : int
        Number of degrees of freedom.
    dx : float
        Step size in the of spatial discretization.
    c : float
        The advective coefficient.

    Returns
    -------
    K : matrix (sparse csr format)
        The time derivative part of the pde obtained from the spatial part.
    b : vector (dense array)
        The remaining term, in this homogeneous case it is a zero array.
    """
    K = -c * Dx(dof, dx)
    b = np.zeros(dof)
    return K, b


def advectivediffusive_fd(dof, dx, mu, c):
    r"""
    Time derivative of the PDE for advective diffusive problems.

    .. math::
        u_{t} + c u_{x} = \mu u_{xx}  \qquad \forall \, x \in \Omega = [0, L] \quad \& \quad t>0

    Thus this returns:

    .. math::
        u_{t} = - c u_{x} + \mu u_{xx}

    Because we use finite difference based matrix products we can convert this
    into a matrix vector product, where :math:`D_x` is the central difference
    approximation of :math:`\partial_x` and similarly  :math:`D_{xx}` the
    central difference approximation of :math:`\partial_{xx}`:

    .. math::
        u_{t} = -c D_{x} u +  \mu D_{xx} u = (-c D_{x} + \mu D_{xx})\, u = K u

    This function calculates the matrix :math:`K`. Because it should be
    compatible with general, non-homogeneous formulation, a part that is
    independent of :math:`u` is also included.

    Parameters
    ----------
    dof : int
        Number of degrees of freedom.
    dx : float
        Step size in the of spatial discretization.
    mu : float
        The diffusive coefficient.
    c : float
        The advective coefficient.

    Returns
    -------
    K : matrix (sparse csr format)
        The time derivative part of the pde obtained from the spatial part.
    b : vector (dense array)
        The remaining term, in this homogeneous case it is a zero array.
    """
    K = -c * Dx(dof, dx) + mu * Dxx(dof, dx)
    b = np.zeros(dof)
    return K, b


def poisson_fd(dof, dx, f, c=1):
    r"""
    Problem formulation of a Poisson equation.

    .. math::
        c u_{xx} = f(x)  \qquad \forall \, x \in \Omega = [0, L]

    Because we use finite difference based matrix products we can convert this
    into a matrix vector product, where :math:`D_{xx}` the is the
    central difference approximation of :math:`\partial_{xx}`:

    .. math::
        D_{xx} u = K u = f/c

    This function calculates the matrix :math:`K` and the forcing vector :math:`f`.
    The matrix is however singular as no boundary conditions are specified.

    Parameters
    ----------
    dof : int
        Number of degrees of freedom.
    dx : float
        Step size in the of spatial discretization.
    f : callable
        A function to calculate the forcing term for any location :math:`x`.
    c : float, optional
        A scalar multiplying the derivative.

    Returns
    -------
    K : matrix (sparse csr format)
        The stiffness matrix.
    b : vector (dense array)
        The right hand side, caused by the non-homogeneous behavior.
    """
    K = Dxx(dof, dx, bc='none')

    # Calculate the right hand side.
    x = dx * np.linspace(0, dof - 1, dof)
    b = f(x) / c
    return K, b


def projection(x, c, fun, num_q, order=1):
    """
    Projecting a 1D function :math:`f(x)` on a finite element basis.

    Lets create our approximation fnuction,

    .. math::
        f_h(x) = \sum_{n=0}^N u_n \phi_n(x)

    as a weighted summation of the basisfunctions of approximation.
    That is all basisfunctions fall within our FE space :math:\phi_n \in V_h`.
    The unknows here are the weights :math:`f_n`.
    To find these weights we formulate a weak form:

    .. math::
        \int_\Omega \qty(f_h(x) - f(x) ) \phi_i dV = 0 \quad \forall \quad \phi_i \in V_h

    in which we substitute our approximation function and separate the knowns from the unknowns.
    We find:

    .. math::
         \int_\Omega \phi_i \sum_{j=0}^N u_j \phi_j(x) dV = \int_\Omega \phi_i f(x) dV \quad \forall \quad \phi\in V_h

    As the weights :math:`u_n` are independent of location, we can take them out of the integral:

    .. math::
        \sum_{j=0}^N u_j int_\Omega \phi_i \phi_j(x) dV = \int_\Omega \phi_i f(x) dV \quad \forall \quad \phi\in V_h

    Which can be rewritten as a system of linear equations, which is:

    .. math::
        M \, u = b

    Where :math:`M` is a matrix and :math:`u,\,b` are vectors.

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
    M, b = kernel1d(x, c, fun, num_q, order, mass=True)

    # Convert matrix objects to sparse counterparts.
    M = sparse.coo_matrix(M).tocsr()
    return M, b
