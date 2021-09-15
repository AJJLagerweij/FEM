r"""
Various implementations of the method of lines to progress through time.
The goal is to implement the code in python and not rely on existing solvers.

Bram Lagerweij
COHMAS Mechanical Engineering KAUST
2021
"""

# Importing External modules
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve


def forwardEuler(pde, u, dt, t_end):
    r"""
    Itterate a through time with the forward Eurler method.

    Lets assume that, through any type of discretization, the time derivative was obtained.
    This time derivative can be represented through linear algabra as:

    .. math::
        M \, u_t = K \, u + b \qquad \text{that is} \qquad u_t = M^{-1}(K u + b)

    where :math:`M` is the mass matrix, :math:`K` the siffness and transport matrix
    and vector :math:`b` the right hand side. these are obtained from approximations
    of the spatial derivatives defined by the functien provided to `func`.

    The backward Euler method predicts the field of our function based upon
    information of the previous timestep only. Imagine that we are at timestep
    :math:`n` and want to predict our field at timestep :math:`u^{(n+1)}`.
    Now a forward finite difference approximation is used:

    .. math::
        u^{(n)}_t = \frac{-u^{(n)} + u^{(n+1)} }{dt}

    That is we can predict our field in the future timestep as:

    .. math::
        u^{(n+1)} = u^{(n)} + dt\, u^{(n)}_t

    Now from our linear algabra implementation we substitute :math:`u_t`

    .. math::
        u^{(n+1)} = u^{(n)} + dt\, M^{-1}(K u^{(n)} + b)

    most important of all is to see that everything on the right hand side is
    exactly known. Thus the updated field can be calculated directly. However
    For this purpouse we would have to invert the mass matrix. If the mass matrix
    is the identity matrix this is simple, but in generally this is not the case.
    As we don't want to invert large matrices, we multiply all terms by :math:`M`.

    .. math::
        M u^{(n+1)} = M u^{(n)} + dt\,(K u^{(n)} + b)

    Which is a system of equations as everything on the right hand side is known and
    can be calculated directly.

    Notes
    -----
    This code will recognize if :math:`M` is the identity matrix and, in that case
    it will solve the problem directly, avoiding the need to solve a sytem of equations.

    Parameters
    ----------
    pde : tuple
        The linear algabra objects of the pde :math:`M\,u_t = K\,u + b`.
    u : array_like
        The field at the start :math:`u(t=0)`.
    dt : float
        The size of the step.
    t_end : float
        Time at termination.

    Returns
    -------
    array_like
        The function for all time steps.
    """
    # Calculate number of timesteps
    max_iter = int(t_end / dt)

    # The t derivative matrix is constant, as it is expensive to
    # build these kind of matrices we make it only once.
    M, K, b = pde

    # Check if M is an identity matrix
    eye = identity(M.shape[0], format='csr')
    if (M!=eye).nnz==0:
        for n in range(max_iter):
            u = u + dt * (K * u + b)

    # Mass matrix is not identity matrix, hence we need
    # to solve a system of equations.
    else:
        for n in range(max_iter):
            rhs = M * u + dt * (K * u + b)
            u = spsolve(M, rhs)

    return u


def backwardEuler(pde, u, dt, t_end):
    r"""
    Itterate a through time with the backward Eurler method.

    Lets assume that, through any type of discretization, the time derivative was obtained.
    This time derivative can be represented through linear algabra as:

    .. math::
        M\,u_t = K\,u + b \qquad \text{that is} \qquad u_t = M^{-1}(K\,u + b)

    where :math:`M` is the mass matrix, :math:`K` the siffness and transport matrix
    and vector :math:`b` the right hand side. these are obtained from approximations
    of the spatial derivatives defined by the functien provided to `func`

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

    in which we substitute the linear algabra representation of our PD.

    .. math::
        u^{(n+1)} = u^{(n)} + dt\, M^{-1}(K u^{n+1} + b)

    It is important to notic that there is a term with an unknown, as that is
    at time step :math:`n+1' on both sides of the equation. Now we rewrite it
    into a system of equations where we find all unknowns on the left hand side
    and all knownn on the right hand side.

    .. math::
        (M - dt\,K)\,u^{(n+1)} = M\,u^{(n)} + dt b

    This is a system of equations which can be solved.

    Parameters
    ----------
    pde : tuple
        The linear algabra objects of the pde :math:`M\,u_t = K\,u + b`.
    u : array_like
        The field at the start :math:`u(t=0)`.
    dt : float
        The size of the time step.
    t_end : float
        Time at termination.

    Returns
    -------
    array_like
        The function for all time steps.
    """
    # Calculate number of timesteps.
    max_iter = int(t_end / dt)

    # The t derivative matrix is constant, as it is expensive to build these
    # kind of matrices we make it only once.
    M, K, b = pde
    A = M - dt*K

    # Update the timesteps with the implicit scheme.
    for n in range(max_iter):
        rhs = M*u + dt*b
        u = spsolve(A, rhs)

    return u


def solve(K, b):
    r"""
    Solve a time independed problem.

    Parameters
    ----------
    func : callable
        The linear algabra problem that we want to solve :math:`K\,u = b`.
    args : tuple, optional
        The parameters into the PDE approximation. Defealts to an empty tuple.

    Returns
    -------
    array_like
        The vector containing :math:`u`.
    """
    # Solvet the system of equations.
    u = spsolve(K, b)
    return u
