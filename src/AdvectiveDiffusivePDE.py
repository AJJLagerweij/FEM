r"""
Solving an Advective and Diffusive PDE with finite differences.

The PDE described by

.. math::
    u_{t} + u_{x} = μu_{xx}  \quad ∀x ∈ Ω = [0, 1]  \;\; \& \;\;  t > 0

Whith a periodic boundary condition. It will show a combination of diffusive
and advective behaviour. The approximation used is a second order finite
difference scheme in space with both a forward and backward Euler method of
lines implementation to handle the time direction.

The goal is to implement the code in python and not rely on existing solvers.

Bram Lagerweij
COHMAS Mechanical Engineering KAUST
2021
"""

# Importing External modules
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def forwardEuler(func, u0, dt, t_end, args=()):
    r"""
    Itterate a through time with the forward Eurler method.

    The backward Euler method predicts the field of our function based upon
    information of the previous timestep only. Imagine that we are at timestep
    :math:`n` and want to predict our field at timestep :math:`u^({n+1})`.
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
    u0 : array_like
        The field at the start :math:`u(t=0)`.
    dt : float
        The size of the step.
    t_end : float
        Time at termination.
    args : tuple, optional
        The parameters into the PDE approximation. Defealts to an empty tuple.

    Returns
    -------
    u : array_like
        The function for all time steps.
    """
    # Prepare array to store the sulution in, if memory becomes an issue we
    # can store only the previous timestep but due to plotting I want to store
    # everything.
    max_iter = int(t_end / dt)
    u = np.zeros((max_iter+1, len(u0)))
    u[0] = u0

    # The t derivative matrix is constant, as it is expensive to build these
    # kind of matrices we make it only once.
    K, b = func(*args)

    # Update the timesteps.
    for n in range(max_iter):
        u[n+1] = u[n] + dt * (K * u[n] + b)
    return u


def backwardEuler(func, u0, dt, t_end, args=()):
    r"""
    Itterate a through time with the backward Eurler method.

    The backward Euler method predicts the field of our function based upon
    information of the previous timestep only. Imagine that we are at timestep
    :math:`n` and want to predict our field at timestep :math:`u^({n+1})`.
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
    u0 : array_like
        The field at the start :math:`u(t=0)`.
    dt : float
        The size of the time step.
    t_end : float
        Time at termination.
    args : tuple, optional
        The parameters into the PDE approximation. Defealts to an empty tuple.

    Returns
    -------
    u : array_like
        The function for all time steps.
    """
    # Prepare array to store the sulution in, if memory becomes an issue we
    # can store only the previous timestep but due to plotting I want to store
    # everything.
    max_iter = int(t_end / dt)
    u = np.zeros((max_iter+1, len(u0)))
    u[0] = u0

    # The t derivative matrix is constant, as it is expensive to build these
    # kind of matrices we make it only once.
    K, b = func(*args)
    A = sparse.identity(dof) - dt*K

    # Update the timesteps with the implicit scheme.
    for n in range(max_iter):
        u[n+1] = spsolve(A, u[n] + dt*b)
    return u


def Dx(dof, dx, bc='periodic'):
    r"""
    Return the central differences matrix for the first derivatie.

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
    D1 : matrix (sparse csr format)
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
    D1 = matrix.tocsr()
    return D1


def Dxx(dof, dx, bc='periodic'):
    r"""
    Return the central differences matrix for the second derivatie.

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
    D2 : matrix (sparse csr format)
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
    D2 = matrix.tocsr()
    return D2


def advectdiffuse(dof, dx, mu):
    r"""
    Time derivative of the PDE for advective diffusive problems.

    .. math::
        u_{t} + u_{x} = μu_{xx}

    Thus this returns:

    .. math::
        u_{t} = - u_{x} + μu_{xx}

    Because we use finite difference based matrix products we can convert this
    into a matrix vector product:

    .. math::
        u_{t} = D^{(1)} u +  μD^{(2)} u = (D^{(1)} + μD^{(2)})\, u = K u

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
        The avective coefficient.

    Returns
    -------
    K : matrix (sparse csr format)
        The time derivative part of the pde obtained from the spatial part.
    b : array_like (1D)
        The remaining term, in this homeneous case it is a zero array.
    """
    K = -Dx(dof, dx) + mu * Dxx(dof, dx)
    b = np.zeros(dof)
    return K, b


def plot_update(i):
    r"""
    Update the matplotlib plot.

    Parameters
    ----------
    i : int
        Iteration number.

    Returns
    -------
    line_forw : lines.line2D
        The updated Forward Euler predicition.
    line_back : lines.line2D
        The updated backwards Euler prediction.
    annotation : text.Annotation
        The annotated text shows the time progression.

    """
    line_forw.set_data(x, u_forw[i])
    line_back.set_data(x, u_back[i])
    annotation.set_text('time t={:.3f}'.format(dt*i))
    return line_forw, line_back, annotation


# Main code, hidden if this file is run as a module.
if __name__ == "__main__":
    # Define properties
    dx = 1e-2
    dt = 5e-3
    t_end = 1
    mu = 0.01
    # euler = 'forward'

    # Define discrete ranges
    dof = int(1/dx) + 1
    x = np.linspace(0, 1, dof)
    dx = x[1] - x[0]
    t = np.arange(0, t_end+dt, step=dt)

    # Prepare solver
    u = np.sin(2*np.pi*x)  # Initial condition

    # Solve the problem using method of lines.
    u_forw = forwardEuler(advectdiffuse, u, dt, t_end, args=(dof, dx, mu))
    u_back = backwardEuler(advectdiffuse, u, dt, t_end, args=(dof, dx, mu))

    # Plotting with animation
    fig, ax = plt.subplots()
    # ax.set_xlim(0, 1)

    line_forw, = ax.plot(x, u_forw[0], label='forward')
    line_back, = ax.plot(x, u_back[0], label='backward')
    annotation = ax.annotate('time t=0', xy=(0.5, 1))

    plt.legend()

    ani = animation.FuncAnimation(fig, plot_update, frames=len(u_forw),
                                  interval=1, blit=True, repeat=False)
