r"""
Finite difference example problems.

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

Bram Lagerweij
COHMAS Mechanical Engineering KAUST
2021
"""

# Importing External modules
import numpy as np
import scipy.sparse as sparse


def Dx(dof, dx, bc='periodic'):
    r"""
    Return the central differences matrix for the first derivative. That is
    the matrix :math:`D_{x}` represents the central difference approximation
    of :math:`\partial_{x}` in 1D axis systems.

    Parameters
    ----------
    dof : int
        Number of spacial degrees of freedom.
    dx : float
        Spacial step size.
    bc : str, optional
        The type of boundary condition to be used. The default is 'periodic'.

    Raises
    ------
    NotImplementedError
        Is raised when the requested boundary condition is not implemented.

    Returns
    -------
    matrix (sparse csr format)
        The central difference approximation of the first derivative.

    Notes
    -----
    The following boundary conditions are possible:

    - 'periodic' (default) that the first and last dofs are representing
      the same point. As a result the derivative of the first point depends
      on the second last point and the derivative of the last point will depend
      on the second point as well.
    - 'none' means that the row of the first and last degree of freedom are
      left empty. This will result in a singular matrix, thus extra constraints
      will have to be defined before solving a system with such a matrix.
    """
    shape = (dof, dof)
    diag = [-1/(2*dx), 1/(2*dx)]
    matrix = sparse.diags(diag, [-1, 1], shape=shape, format='lil')

    if bc == 'periodic':  # remover first column and last row
        matrix[0, -2] = -1 / (2*dx)
        matrix[-1, 1] = 1 / (2*dx)
    elif bc == 'none':  # use 1st order finite at the start and end.
        matrix[0, 0] = 0
        matrix[0, 1] = 0
        matrix[-1, -2] = 0
        matrix[-1, -1] = 0
    else:
        raise NotImplementedError("This type of bounary condition is not recognized.")
    return matrix.tocsr()


def Dxx(dof, dx, bc='periodic'):
    r"""
    Return the central differences matrix for the second derivative. That is
    the matrix :math:`D_{xx}` represents the central difference approximation
    of :math:`\partial_{xx}` in 1D axis systems.

    Parameters
    ----------
    dof : int
        Number of spacial degrees of freedom.
    dx : float
        Spacial step size.
    bc : str, optional
        The type of boundary condition to be used. The default is 'periodic'.

    Raises
    ------
    NotImplementedError
        Is raised when the requested boundary condition is not implemented.

    Returns
    -------
    matrix (sparse csr format)
        The central difference approximation of the first derivative.

    Notes
    -----
    The following boundary conditions are possible:

    - 'periodic' (defeat) that the first and last dofs are representing
      the same point. As a result the derivative of the first point depends
      on the second last point and the derivative of the last point will depend
      on the second point as well.
    - 'none' means that the row of the first and last degree of freedom are
      left empty. This will result in a singular matrix, thus extra constraints
      will have to be defined before solving a system with such a matrix.

    """
    shape = (dof, dof)
    diag = [1 / (dx**2), -2 / (dx**2), 1 / (dx**2)]
    matrix = sparse.diags(diag, [-1, 0, 1], shape=shape, format='lil')

    if bc == 'periodic':  # remover first column and last row
        matrix[0, -2] = 1 / (dx**2)
        matrix[-1, 1] = 1 / (dx**2)
    elif bc == 'none':  # use 1st order finite at the start and end.
        matrix[0, 0] = 0
        matrix[0, 1] = 0
        matrix[-1, -2] = 0
        matrix[-1, -1] = 0
    else:
        raise NotImplementedError("This type of bounary condition is not recognized.")
    return matrix.tocsr()


def diffusive(dof, dx, mu):
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
    M : matrix (sparse csr format)
        The mass matrix, which will equal the identity matrix in finite differenc problems.
    K : matrix (sparse csr format)
        The time derivative part of the pde obtained from the spatial part.
    b : vector (dense array)
        The remaining term, in this homogeneous case it is a zero array.
    """
    M = sparse.identity(dof, format='csr')
    K = mu * Dxx(dof, dx)
    b = np.zeros(dof)
    return M, K, b


def advective(dof, dx, c):
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
    M : matrix (sparse csr format)
        The mass matrix, which will equal the identity matrix in finite differenc problems.
    K : matrix (sparse csr format)
        The time derivative part of the pde obtained from the spatial part.
    b : vector (dense array)
        The remaining term, in this homogeneous case it is a zero array.
    """
    M = sparse.identity(dof, format='csr')
    K = -c * Dx(dof, dx)
    b = np.zeros(dof)
    return M, K, b


def advectivediffusive(dof, dx, mu, c):
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
    M : matrix (sparse csr format)
        The mass matrix, which will equal the identity matrix in finite differenc problems.
    K : matrix (sparse csr format)
        The time derivative part of the pde obtained from the spatial part.
    b : vector (dense array)
        The remaining term, in this homogeneous case it is a zero array.
    """
    M = sparse.identity(dof, format='csr')
    K = -c * Dx(dof, dx) + mu * Dxx(dof, dx)
    b = np.zeros(dof)
    return M, K, b


def poisson(dof, dx, f, c=1):
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
