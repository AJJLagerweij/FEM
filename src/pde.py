r"""
Storing varios PDE's that can be will be solved in this course. This includes:

- Diffusive 1D

.. math::
	u_{t} = \mu u_{xx} \qquad \forall \, x \in \Omega = [0, 1] \quad \& \quad t>0

- Advective 1D

.. math::
	u_{t} + c {u}_{x} = 0 \qquad \forall \, x \in \Omega = [0, 1] \quad \& \quad t>0

- Diffusive-Advective 1D

.. math::
	u_{t} + c {u}_{x} = \mu u_{xx} \qquad \forall \, x \in \Omega = [0, 1] \quad \& \quad t>0

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


def diffusive(dof, dx, mu):
    r"""
    Time derivative of the PDE for advective diffusive problems.
    AA

    .. math::
        u_{t} = \mu u_{xx}  \qquad \forall \, x \in \Omega = [0, 1] \quad \& \quad t>0

    Thus this returns:

    .. math::
        u_{t} = \mu u_{xx}

    Because we use finite difference based matrix products we can convert this
    into a matrix vector product, where :math:`D_xx` is the central difference
    approximation of :math:`\partial_xx`:

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
        Stepsize in the of spatial discretisation.
    mu : float
        The defusive coefficient.

    Returns
    -------
    K : matrix (sparse csr format)
        The time derivative part of the pde obtained from the spatial part.
    b : vector (dense array)
        The remaining term, in this homeneous case it is a zero array.
    """
    K = mu * Dxx(dof, dx)
    b = np.zeros(dof)
    return K, b

def advective(dof, dx, c):
    r"""
    Time derivative of the PDE for advective diffusive problems.

    .. math::
        u_{t} + c u_{x} = 0  \qquad \forall \, x \in \Omega = [0, 1] \quad \& \quad t>0

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
        Stepsize in the of spatial discretisation.
    c : float
        The avective coefficient.

    Returns
    -------
    K : matrix (sparse csr format)
        The time derivative part of the pde obtained from the spatial part.
    b : vector (dense array)
        The remaining term, in this homeneous case it is a zero array.
    """
    K = -c * Dx(dof, dx)
    b = np.zeros(dof)
    return K, b



def advectivediffusive(dof, dx, mu, c):
    r"""
    Time derivative of the PDE for advective diffusive problems.

    .. math::
        u_{t} + c u_{x} = \mu u_{xx}  \qquad \forall \, x \in \Omega = [0, 1] \quad \& \quad t>0

    Thus this returns:

    .. math::
        u_{t} = - c u_{x} + \mu u_{xx}

    Because we use finite difference based matrix products we can convert this
    into a matrix vector product, where :math:`D_x` is the central difference
    approximation of :math:`\partial_x` and similarly  :math:`D_{xx}` the
    central difference apprximation of :math:`\partial_{xx}`:

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
        Stepsize in the of spatial discretisation.
    mu : float
        The diffusive coefficient.
    c : float
    	The avective coefficient.

    Returns
    -------
    K : matrix (sparse csr format)
        The time derivative part of the pde obtained from the spatial part.
    b : vector (dense array)
        The remaining term, in this homeneous case it is a zero array.
    """
    K = -c * Dx(dof, dx) + mu * Dxx(dof, dx)
    b = np.zeros(dof)
    return K, b

