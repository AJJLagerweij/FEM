r"""
Storing various PDEs that can be will be solved in this course. This includes:

- Diffusive 1D

    .. math::
       u_{t} - \mu u_{xx} = 0 \qquad \forall \, x \in \Omega = [0, L] \quad \& \quad t>0

- Advective 1D

    .. math::
       u_{t} + c {u}_{x} = 0 \qquad \forall \, x \in \Omega = [0, L] \quad \& \quad t>0

- Diffusive-Advective 1D

    .. math::
       u_{t} + c {u}_{x} - \mu u_{xx} = 0 \qquad \forall \, x \in \Omega = [0, L] \quad \& \quad t>0

- Poisson in 1D

    .. math::
       u_{xx} = f(x) \qquad \forall \, x \in \Omega = [0, L]

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


def projection(x, connect, fun, num_q, order):
    r"""
    Projecting a 1D function :math:`f(x)` on a finite element basis.

    Lets create our approximation function,

    .. math::
        f_h(x) = \sum_{n=0}^N \bar{u}_n \phi_n(x)

    as a weighted summation of the basisfunctions of approximation. Where :math:`phi_n` are the basisfunctions of our
    FE space :math:`V_h`. The unknows here are the weights :math:`\bar{u}_n`, these we call degrees of freedom.
    To find these DOFs we formulate a weak form:

    .. math::
        \int_\Omega (f_h(x) - f(x)) \, \phi_i(x)\, dV = 0 \quad \forall \quad \phi_i \in V_h

    in which we substitute our approximation function and separate the knowns from the unknowns. We find:

    .. math::
         \int_\Omega \phi_i(x) \sum_{j=0}^N \bar{u}_j \phi_j(x)\, dV =
         \int_\Omega \phi_i(x) f(x) \, dV \quad \forall \quad \phi_i \in V_h

    As the weights :math:`\bar{u}_n` are independent of location, we can take them out of the integral:

    .. math::
        \sum_{j=0}^N \bar{u}_j \int_\Omega \phi_i(x) \phi_j(x)\, dV =
        \int_\Omega \phi_i(x) f(x)\, dV \quad \forall \quad \phi_i \in V_h

    Which can be rewritten as a system of linear equations, which is:

    .. math::
        M \, \bar{u} = b

    Where :math:`M` is a matrix and :math:`\bar{u}` and :math:`b` are vectors.

    Parameters
    ----------
    x : array_like(float)
        Global coordinates of all degrees of freedom.
    connect : array_like(int), shape((num_ele, dofe/ele))
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
    M, T, S, b = kernel1d(x, connect, fun, num_q, order, mass=True)

    # Convert matrix objects to sparse counterparts.
    M = sparse.coo_matrix(M).tocsr()
    return M, b


def diffusive(x, connect, mu, num_q, order):
    r"""
    Time derivative of the PDE for diffusion problems.

    .. math::
        u_{t} - \mu u_{xx} = 0 \qquad \forall \, x \in \Omega = [0, L] \quad \& \quad t>0

    Which, is converted into a weak form through:

    .. math::
        \int_\Omega (\tilde{u}_t - \mu \tilde{u}_{xx} ) \phi_i(x) dV = 0 \quad \forall \quad \phi_i \in V_h

    Where :math:`\tilde{u}` is our approximation:

    .. math::
        \tilde{u}(x) = \sum_{n = 1}^N \bar{u}_n \phi_n(x)

    in which :math:`\bar{u}_n` are the unknowns and :math:`\phi_n(x)` are the basisfunctions of
    FE approximation space :math:`V_h`. Substituting this approximation leads to:

    .. math::
        \int_\Omega (\partial_t\sum_{j = 1}^N \bar{u}_j \phi_j(x) - \mu \partial_{xx}\sum_{j = 1}^N \bar{u}_j \phi_j(x) ) \phi_i(x)
        dV = 0 \quad \forall \quad \phi_i \in V_h

    Which we split into different integrals:

    .. math::
        \int_\Omega  (\partial_t \sum_{j = 1}^N \bar{u}_j \phi_j(x))\phi_i(x) dV -
        \int_\Omega  (\mu\partial_{xx}\sum_{j = 1}^N \bar{u}_j \phi_j(x)) \phi_i(x) dV = 0 \quad \forall \quad \phi_i \in V_h

    For the first integral we notice that the basis functions are constant through time, only the degrees of freedom
    :math:`\bar{u}_j` vary through time. Similarly these degrees of freedom does not affect the integral over space
    :math:`\int_\Omega dV`. Thus we can write:

    .. math::
        \int_\Omega (\partial_t \sum_{j = 1}^N \bar{u}_j \phi_j(x)) \phi_i(x) dV =
        \sum_{j = 1}^N \partial_t \bar{u}_j \int_\Omega \phi_j(x) \phi_i(x) dV = M \bar{u}_j

    where :math:`M` is the mass matrix which combines the integral for all different basis functions.
    For the second term we apply integration by parts, while assuming that Diriclet boundary conditions:

    .. math::
        \int_\Omega (\mu \partial_{xx}\sum_{j = 1}^N \bar{u}_j \phi_j(x)) \phi_i(x) dV =
        \int_\Omega (\mu \partial_x\sum_{j = 1}^N \bar{u}_j \phi_j(x)) \partial_x\phi_i(x) dV =
        \mu \sum_{j=1}^N \bar{u}_j \int_\Omega \partial_x \phi_j(x) \partial_x\phi_i(x) dV = \mu S \bar{u}_j

    where :math:`S` is the stiffness matrix, which can be computed independently from the actual unknowns.
    Now we can write our PDE in terms of linear algabra objects:

    .. math::
        M \bar{u}_t - \mu S \bar{u} = 0

    which we modify to be in the format is expected by the temporal solvers:

    .. math::
        M \bar{u}_t = K \bar{u}

    Parameters
    ----------
    x : array_like(float)
        Global coordinates of all degrees of freedom.
    connect : array_like(int), shape((num_ele, dofe/ele))
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
    K : matrix, (sparse csr format)
        The stiffeness matrix scaled with the diffusivity constant :math:`K = \mu S`.
    b : vector, (dense array)
        The right hand side, because we consider a homogeneous PDE with diriclet conditions it is a zero vector.
    """
    # Obtain matrix elements from main FEM kernel.
    M, T, S, b = kernel1d(x, connect, None, num_q, order, mass=True, stiffness=True)

    # Convert matrix objects to sparse counterparts.
    M = sparse.coo_matrix(M).tocsr()
    K = mu * sparse.coo_matrix(S).tocsr()
    return M, K, b


def advective(x, connect, c, num_q, order):
    r"""
    Time derivative of the PDE for advective diffusive problems.

    .. math::
        u_{t} + c u_x = 0 \qquad \forall \, x \in \Omega = [0, L] \quad \& \quad t>0

    Which, is converted into a weak form through:

    .. math::
        \int_\Omega (\tilde{u}_t + c \tilde{u}_x ) \phi_i(x) dV =
        0 \quad \forall \quad \phi_i \in V_h

    Where :math:`\tilde{u}` is our approximation:

    .. math::
        \tilde{u}(x) = \sum_{n = 1}^N \bar{u}_n \phi_n(x)

    in which :math:`\bar{u}_n` are the unknowns, socalled degrees of freedom and :math:`\phi_n(x)` are the
    basisfunctions of FE approximation space :math:`V_h`. Substituting this approximation leads to:

    .. math::
        \int_\Omega (\partial_t\sum_{j = 1}^N \bar{u}_j \phi_j(x) +
        c \partial_x \sum_{j = 1}^N \bar{u}_j \phi_j(x)) \phi_i(x) dV
        = 0 \quad \forall \quad \phi_i \in V_h

    Which we split into different integrals:

    .. math::
        \int_\Omega  (\partial_t \sum_{j = 1}^N \bar{u}_j \phi_j(x))\phi_i(x) dV +
        \int_\Omega  (c\partial_x\sum_{j = 1}^N \bar{u}_j \phi_j(x)) \phi_i(x) dV =
        0 \quad \forall \quad \phi_i \in V_h

    For the first integral we notice that the basis functions are constant through time, only the degrees of freedom
    :math:`\bar{u}_j` vary through time. Similarly these degrees of freedom does not affect the integral over space
    :math:`\int_\Omega dV`. Thus we can write:

    .. math::
        \int_\Omega (\partial_t \sum_{j = 1}^N \bar{u}_j \phi_j(x)) \phi_i(x) dV =
        \sum_{j = 1}^N \partial_t \bar{u}_j \int_\Omega \phi_j(x) \phi_i(x) dV = M \bar{u}_j

    where :math:`M` is the mass matrix which combines the integral for all different basis functions.
    For the second term we aknoledge that the degrees of freedom have no spatial and temporal effects thus
    we can take them out of the integral and derivatives.

    .. math::
        \int_\Omega  (c\partial_x\sum_{j = 1}^N \bar{u}_j \phi_j(x)) \phi_i(x) dV =
        c \sum_{j = 1}^N \bar{u}_j \int_\Omega \partial_x\phi_j(x) \phi_i(x) dV = c T \bar{u}

    where :math:`T` is the socalled transport matrix, wich only depends on the basis functions.

    Now we can write our PDE in terms of linear algabra objects:

    .. math::
        M \bar{u}_t + c T \bar{u} = 0

    which we modify to be in the format is expected by the temporal solvers:

    .. math::
        M \bar{u}_t = K \bar{u}

    Parameters
    ----------
    x : array_like(float)
        Global coordinates of all degrees of freedom.
    connect : array_like(int), shape((num_ele, dofe/ele))
        Element to degree of freedom connectivety map.
    c : float
        Advective constant.
    num_q : int
        Number of Gausian quadrature points.
    order : int
        Order of the polynomial used by our element.

    Returns
    -------
    M : matrix, (sparse csr format)
        The mass matrix.
    K : matrix, (sparse csr format)
        The combination of stiffness and transport matrix matrix scaled with the approprate constants
         :math:`K = - c T`.
    b : vector, (dense array)
        The right hand side, because we consider a homogeneous PDE with diriclet conditions it is a zero vector.
    """
    # Obtain matrix elements from main FEM kernel.
    M, T, S, b = kernel1d(x, connect, None, num_q, order, mass=True, transport=True)

    # Convert matrix objects to sparse counterparts.
    M = sparse.coo_matrix(M).tocsr()
    K = - c * sparse.coo_matrix(T).tocsr()
    return M, K, b


def advectivediffusive(x, connect, c, mu, num_q, order):
    r"""
    Time derivative of the PDE for advective diffusive problems.

    .. math::
        u_{t} + c u_x - \mu u_{xx} = 0 \qquad \forall \, x \in \Omega = [0, L] \quad \& \quad t>0

    Which, is converted into a weak form through:

    .. math::
        \int_\Omega (\tilde{u}_t + c \tilde{u}_x - \mu \tilde{u}_{xx} ) \phi_i(x) dV =
        0 \quad \forall \quad \phi_i \in V_h

    Where :math:`\tilde{u}` is our approximation:

    .. math::
        \tilde{u}(x) = \sum_{n = 1}^N \bar{u}_n \phi_n(x)

    in which :math:`\bar{u}_n` are the unknowns, socalled degrees of freedom and :math:`\phi_n(x)` are the
    basisfunctions of FE approximation space :math:`V_h`. Substituting this approximation leads to:

    .. math::
        \int_\Omega (\partial_t\sum_{j = 1}^N \bar{u}_j \phi_j(x) +
        c \partial_x \sum_{j = 1}^N \bar{u}_j \phi_j(x) -
        \mu \partial_{xx}\sum_{j = 1}^N \bar{u}_j \phi_j(x) ) \phi_i(x) dV
        = 0 \quad \forall \quad \phi_i \in V_h

    Which we split into different integrals:

    .. math::
        \int_\Omega  (\partial_t \sum_{j = 1}^N \bar{u}_j \phi_j(x))\phi_i(x) dV +
        \int_\Omega  (c\partial_x\sum_{j = 1}^N \bar{u}_j \phi_j(x)) \phi_i(x) dV -
        \int_\Omega  (\mu\partial_{xx}\sum_{j = 1}^N \bar{u}_j \phi_j(x)) \phi_i(x) dV =
        0 \quad \forall \quad \phi_i \in V_h

    For the first integral we notice that the basis functions are constant through time, only the degrees of freedom
    :math:`\bar{u}_j` vary through time. Similarly these degrees of freedom does not affect the integral over space
    :math:`\int_\Omega dV`. Thus we can write:

    .. math::
        \int_\Omega (\partial_t \sum_{j = 1}^N \bar{u}_j \phi_j(x)) \phi_i(x) dV =
        \sum_{j = 1}^N \partial_t \bar{u}_j \int_\Omega \phi_j(x) \phi_i(x) dV = M \bar{u}_j

    where :math:`M` is the mass matrix which combines the integral for all different basis functions.
    For the second term we aknoledge that the degrees of freedom have no spatial and temporal effects thus
    we can take them out of the integral and derivatives.

    .. math::
        \int_\Omega  (c\partial_x\sum_{j = 1}^N \bar{u}_j \phi_j(x)) \phi_i(x) dV =
        c \sum_{j = 1}^N \bar{u}_j \int_\Omega \partial_x\phi_j(x) \phi_i(x) dV = c T \bar{u}

    where :math:`T` is the socalled transport matrix, wich only depends on the basis functions.
    For the thrird part we apply integration by parts, while assuming that Diriclet boundary conditions:

    .. math::
        \int_\Omega (\mu \partial_{xx}\sum_{j = 1}^N \bar{u}_j \phi_j(x)) \phi_i(x) dV =
        \int_\Omega (\mu \partial_x\sum_{j = 1}^N \bar{u}_j \phi_j(x)) \partial_x\phi_i(x) dV =
        \mu \sum_{j=1}^N \bar{u}_j \int_\Omega \partial_x \phi_j(x) \partial_x\phi_i(x) dV = \mu S \bar{u}_j

    where :math:`S` is the stiffness matrix, which can be computed independently from the actual unknowns.
    Now we can write our PDE in terms of linear algabra objects:

    .. math::
        M \bar{u}_t + c T \bar{u} - \mu S \bar{u} = 0

    which we modify to be in the format is expected by the temporal solvers:

    .. math::
        M \bar{u}_t = K \bar{u}

    Parameters
    ----------
    x : array_like(float)
        Global coordinates of all degrees of freedom.
    connect : array_like(int), shape((num_ele, dofe/ele))
        Element to degree of freedom connectivety map.
    c : float
        Advective constant.
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
    K : matrix, (sparse csr format)
        The combination of stiffness and transport matrix matrix scaled with the approprate constants
         :math:`K = \mu S - c T`.
    b : vector, (dense array)
        The right hand side, because we consider a homogeneous PDE with diriclet conditions it is a zero vector.
    """
    # Obtain matrix elements from main FEM kernel.
    M, T, S, b = kernel1d(x, connect, None, num_q, order, mass=True, transport=True, stiffness=True)

    # Convert matrix objects to sparse counterparts.
    M = sparse.coo_matrix(M).tocsr()
    K = mu * sparse.coo_matrix(S).tocsr() - c * sparse.coo_matrix(T).tocsr()
    return M, K, b
