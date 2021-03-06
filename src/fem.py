"""
The main FEM loop.

This main FEM loop will assample the differenc matrices that are required in a FEM solver.
It takes the following steps:

1. Loop over all elements.
2. Compute the element based integrals.
    1. calculate the quantities in the reference element.
    2. integrate using Quadrature rules and include the mapping from reference to global axis system.
3. Assamble these element based contributions into a global operator.

Bram Lagerweij
COHMAS Mechanical Engineering KAUST
2021
"""

# Import external modules.
import numpy as np
import numba as nb

# From my own scripts import
from element import shape1d, get_element, element_rhs, element_mass


@nb.jit(nopython=False)
def kernel1d(x, c, rhs, num_q, order, mass=False):
    """
    Create the global FEM system by looping over the elements.

    Parameters
    ----------
    x : array_like(float)
        Global coordinates of all degrees of freedom.
    c : array_like(int), shape((num_ele, dofe/ele))
        Element to degree of freedom connectivety map.
    rhs : callable
        Function that acts as our right hand side (nonhomogeneous term).
    num_q : int
        Number of Gausian quadrature points.
    order : int
        Order of the polynomial used by our element.
    mass : bool, optional
        Return a mass matrix. Default is `False`.

    Returns
    -------
    f : array_like(float), shape(dofe)
        Global right hand side in our system of equations.
        Only when `rhs != None`, `None` otherwise.
    m : COO (value (row, value))
        Global mass matrix, ready to be converted to COO. Repeating idicess do exist.
        Only `mass == True`, `None` otherwise.
    """
    num_ele = len(c)
    num_dofs = np.max(c) + 1
    num_dofe = len(c[0])  # number of element dofe

    # Initialize right hand side vector (dense).
    if rhs != None:
        f = np.zeros(num_dofs)
    else:
        f = None

    # Initialize mass matrix (sparse coo format).
    if mass is True:
        m_v = np.zeros((num_ele, num_dofe**2))  # Value list
        m_i = np.zeros((num_ele, num_dofe**2), dtype=nb.types.uint)  # Row list
        m_j = np.zeros((num_ele, num_dofe**2), dtype=nb.types.uint)  # Column list
    else:
        m_v, m_i, m_j = None, None, None

    for ele in range(num_ele):
        # Obtain element properties
        dofe = c[ele]
        x_ele = x[dofe]
        phi_xq, wq_phi_xq_detJ, f_xq = get_element(num_q, x_ele, rhs, order=order)

        # Perform integration and compute right hand side vector.
        if rhs != None:
            fe = element_rhs(wq_phi_xq_detJ, f_xq)
            f[dofe] += fe

        # Calculate element mass matrix and add to value colunm row arrays.
        if mass == True:
            me = element_mass(phi_xq, wq_phi_xq_detJ)
            ie = np.repeat(dofe, len(dofe))
            je = ie.reshape((-1, len(dofe))).T.ravel()
            m_v[ele] = me.ravel()
            m_i[ele] = ie
            m_j[ele] = je

    # Flatten mass COO arrays, repeating indices remain.
    if mass == True:
        m = (m_v.ravel(), (m_i.ravel(), m_j.ravel()))
    else:
        m = None
    return m, f


