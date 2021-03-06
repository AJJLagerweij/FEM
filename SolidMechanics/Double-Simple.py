r"""
Solving a linear Euler-Bernoulli beam problem with finite differences.

The PDE described by

.. math::
    EI u''(x) = M(x) &  \quad \forall x \in\Omega = [0, L]\\
    \text{where:} \quad & M(x) =  \begin{cases} -3Px & 0 \leq x \leq L/4\\ P(x - L) & L/4 \leq x \leq L \end{cases}\\
    \text{s.t.:} \quad &u(0) = 0 \\
    & u'(0) = 0

Where :math:`P` is the load applied at the tip of our beam. This beam has a length :math:`L` and a stiffness EI.
In general these kinds of problems can not be solved directly in this way, as it is not always possible to describe
the moment explicitly, but because our cantilever beam is statically determinate and our material function invertible
The problem is a Poisson equation, a non-homogeneous but linear first order ODE.
Which we will be solved using a central difference approximation

Bram Lagerweij
COHMAS Mechanical Engineering KAUST
2021
"""

# Importing External modules
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import cnames
import numpy as np
from scipy.sparse.linalg import spsolve

# Importing my own scripts
sys.path.insert(1, '../src')
from finitedifference import poisson


def moment(P, L):
    """
    Moment as a function of :math:`x` of the double simply supported beam.

    Parameters
    ----------
    P : float
        Applied load.
    L : float
        Length of the beam

    Returns
    -------
    callable
        The moment :math:`M(x)` of the beam.
    """

    def fun(x):
        shape = np.shape(x)

        if len(shape) == 0:
            if x < L / 4:
                m = -3 * P * x
            else:
                m = P * (x - L)
        else:
            m = np.zeros_like(x)
            ind = np.where(x < L / 4)  # where x < L/4
            m[ind] = -3 * P * x[ind]

            ind = np.where(L / 4 <= x)  # where L/4 < x
            m[ind] = P * (x[ind] - L)
        return m

    return fun


if __name__ == '__main__':
    # Define properties of the problem.
    L = 1000  # 1000 mm length
    P = 1  # 1 N load
    EI = 187500000  # Beam bending stiffness Nmm^4

    # Discretion of the space.
    dof = 101  # Number of nodes
    x, dx = np.linspace(0, L, dof, retstep=True)

    # Exact solution (solved by hand).
    u_exact = np.zeros_like(x)
    index = np.where(x <= 250)
    u_exact[index] = P / EI * (31.25e6 * (x[index] / 1000) - 500e6 * (x[index] / 1000) ** 3)
    index = np.where(250 < x)
    u_exact[index] = P / EI * (-62.5e6 * (x[index] / 1000 - 0.25)
                               - 375e6 * (x[index] / 1000 - 0.25) ** 2
                               + 500e6 / 3 * (x[index] / 1000 - 0.25) ** 3)

    # Calculate the internal Moment.
    M = moment(P, L)  # Create a callable for the moment in Nmm

    # Create linear problem.
    K, f = poisson(dof, dx, M, c=EI)

    # Boundary condition u(0) = 0
    K[0, 0] = 1
    f[0] = 0

    # Boundary condition u(L/4) = 0  For this purpose we use
    # the last row of the matrix, this row is not yet used.
    index = int(dof / 4)
    K[-1, index] = 1
    f[-1] = 0

    # Solve the problem.
    u = spsolve(K, f)

    # Plotting the results.
    plt.figure(figsize=(8, 4))
    plt.xlim(0, 1020)
    plt.xlabel("$x$ location")
    lines = []

    c1 = cnames['blue']
    c2 = cnames['dodgerblue']
    c3 = cnames['green']

    ax_u = plt.gca()
    ax_u.set_ylabel('$u(x)$ in mm')
    ax_u.yaxis.label.set_color(c1)
    ax_u.tick_params(axis='y', colors=c1)
    lines += ax_u.plot(x, u_exact, c=c1, label="Displacement Exact")
    lines += ax_u.plot(x, u, ':+', c=c2, label="Displacement")

    ax_m = plt.twinx(ax_u)
    ax_m.set_ylim(-800, 0)
    ax_m.set_ylabel("$M(x)$ in Nmm")
    ax_m.yaxis.label.set_color(c3)
    ax_m.spines['left'].set_color(c1)
    ax_m.spines['right'].set_color(c3)
    ax_m.tick_params(axis='y', colors=c3)
    lines += ax_m.plot(x, M(x), c=c3, label="Moment exact")

    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, bbox_to_anchor=(0., 1.02, 1., .102),
               loc='center', ncol=3, mode='expand')

    ax_u.annotate(r'$P$', xy=(1000, -1), xytext=(1000, -0.75),
                  ha='center', arrowprops=dict(arrowstyle='-|>', lw=1))
    t1 = plt.Polygon([[0, 0], [12, -0.05], [-12, -0.05]], closed=False, color='k')
    ax_u.add_patch(t1)
    t2 = plt.Polygon([[250, 0], [262, -0.05], [238, -0.05]], closed=False, color='k')
    ax_u.add_patch(t2)
    plt.tight_layout()
    plt.show()
