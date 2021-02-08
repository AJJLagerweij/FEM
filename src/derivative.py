"""
Storing various derivatives for the purpose of  importing them into the partial
derivative equations in another scrcipt.

Bram Lagerweij
COHMAS Mechanical Engineering KAUST
2021
"""

# Importing External modules
import numpy as np
import scipy.sparse as sparse


def Dx(dof, dx, bc='periodic'):
    r"""
    Return the central differences matrix for the first derivatie. That is
    the matrix :math:`D_{x}` represents the central difference approximation
    of :math:`\partial_{x}` in 1D axis systems.

    Parameters
    ----------
    dof : int
        Number of spacial degrees of freedom.
    dx : float
        Spacial step size.
    bc : string, optional
        The type of boundary condition to be used. The default is 'periodic'.

    Raises
    ------
    ValueError
        Is raised when the requested boundary condition is not implemented.

    Returns
    -------
    matrix (sparse csr format)
        The central difference approximation of the first derivative.
    """
    shape = (dof, dof)
    diag = [-1/(2*dx), 1/(2*dx)]
    matrix = sparse.diags(diag, [-1, 1], shape=shape, format='lil')

    if bc == 'periodic':  # remover first column and last row
        matrix[0, -2] = -1 / (2*dx)
        matrix[-1, 1] = 1 / (2*dx)
    else:
        raise ValueError("This type of bounary condition is not recognized.")
    return matrix.tocsr()


def Dxx(dof, dx, bc='periodic'):
    r"""
    Return the central differences matrix for the second derivatie. That is
    the matrix :math:`D_{xx}` represents the central difference approximation
    of :math:`\partial_{xx}` in 1D axis systems.

    Parameters
    ----------
    dof : int
        Number of spacial degrees of freedom.
    dx : float
        Spacial step size.
    bc : string, optional
        The type of boundary condition to be used. The default is 'periodic'.

    Raises
    ------
    ValueError
        Is raised when the requested boundary condition is not implemented.

    Returns
    -------
    matrix (sparse csr format)
        The central difference approximation of the first derivative.
    """
    shape = (dof, dof)
    diag = [1 / (dx**2), -2 / (dx**2), 1 / (dx**2)]
    matrix = sparse.diags(diag, [-1, 0, 1], shape=shape, format='lil')

    if bc == 'periodic':  # remover first column and last row
        matrix[0, -2] = 1 / (dx**2)
        matrix[-1, 1] = 1 / (dx**2)
    else:
        raise ValueError("This type of bounary condition is not recognized.")
    return matrix.tocsr()

