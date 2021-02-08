r"""
Various implementations of the method of lines to progress through time.
The goal is to implement the code in python and not rely on existing solvers.

Bram Lagerweij
COHMAS Mechanical Engineering KAUST
2021
"""

# Importing External modules
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve


def forwardEuler(func, u, dt, t_end, args=()):
    r"""
    Itterate a through time with the forward Eurler method.

    The backward Euler method predicts the field of our function based upon
    information of the previous timestep only. Imagine that we are at timestep
    :math:`n` and want to predict our field at timestep :math:`u^{(n+1)}`.
    Now a forward finite difference approximation is used:

    .. math::
        u^{(n)}_t = \frac{-u^{(n)} + u^{(n+1)} }{dt}

    That is we can predict our field in the future timestep as:

    .. math::
        u^{(n+1)} = u^{(n)} + dt\, u^{(n)}_t

    Our time derivative at the current timestep, :math:`u^{(n)}_t` is obtained
    with:

    .. math::
        u_t = K u + b

    where matrix :math:`K` and vector :math:`b` stem from approximations of our
    spatial derivatives defined by the functien provided to `func`. Resulting
    in the following update scheme:

    .. math::
        u^{(n+1)} = u^{(n)} + dt\, (K u^{(n)} + b)

    most important of all is to see that everything on the right hand side is
    exactly known. Thus the updated field can be calculated directly.

    Parameters
    ----------
    func : callable
        The time derivative of the pde to be solved such that :math:`u_t = K\,u + b`.
    u : array_like
        The field at the start :math:`u(t=0)`.
    dt : float
        The size of the step.
    t_end : float
        Time at termination.
    args : tuple, optional
        The parameters into the PDE approximation. Defealts to an empty tuple.

    Returns
    -------
    array_like
        The function for all time steps.
    """
    # The t derivative matrix is constant, as it is expensive to build these
    # kind of matrices we make it only once.
    K, b = func(*args)

    # Update the timesteps.
    max_iter = int(t_end / dt)
    for n in range(max_iter):
        u = u + dt * (K * u + b)
    return u


def backwardEuler(func, u, dt, t_end, args=()):
    r"""
    Itterate a through time with the backward Eurler method.

    The backward Euler method predicts the field of our function based upon
    information of the previous timestep only. Imagine that we are at timestep
    :math:`n` and want to predict our field at timestep :math:`u^{(n+1)}`.
    Now a backward finite difference approximation used the time derivative
    of the next timestep, wich is not yet known:

    .. math::
        u^{(n+1)}_t = \frac{ -u^{(n)} + u^{(n+1)} }{dt}

    That is we can predict our field in the future timestep as:

    .. math::
        u^{(n+1)} = u^{(n)} + dt\, u^{(n+1)}_t

    It is important to notic that there is a term with an unknown, as that is
    at time step :math:`n+1' on both sides of the equation.
    Our time derivative is obtained with an approximation equation:

    .. math::
        u_t = K u + b

    where matrix :math:`K` and vector :math:`b` stem from approximations of our
    spatial derivatives defined by the functien provided to `func`. This
    results in:

    .. math::
        u^{(n+1)} = u^{(n)} + dt\, ( K u^{(n+1)} + b )

    Now we rewrite it into a system of equations where we find all unknowns
    on the left hand side and all knownn on the right hand side.

    .. math::
        (I - dt\,K)\, u^{(n+1)} = u^{(n)} + dt\,b

    Parameters
    ----------
    func : callable
        The time derivative of the pde to be solved such that :math:`u_t = K\,u + b`.
    u : array_like
        The field at the start :math:`u(t=0)`.
    dt : float
        The size of the time step.
    t_end : float
        Time at termination.
    args : tuple, optional
        The parameters into the PDE approximation. Defealts to an empty tuple.

    Returns
    -------
    array_like
        The function for all time steps.
    """
    # The t derivative matrix is constant, as it is expensive to build these
    # kind of matrices we make it only once.
    K, b = func(*args)
    A = sparse.identity(len(b)) - dt*K

    # Update the timesteps with the implicit scheme.
    max_iter = int(t_end / dt)
    for n in range(max_iter):
        u = spsolve(A, u + dt*b)
    return u
