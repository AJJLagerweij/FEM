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


@nb.jit(nopython=True)
def element_mass(phi_xq, wq_detJ):
    r"""
    Compute the elmement mass matrix.

    This matrix is defined as:

    .. math::
        M &= \int_{\Omega} \phi_j(x) \,\, \phi_i(x) \,\, dV  \quad \forall \quad \phi_i, \phi_j \in V_h \\
        &= \sum_{e=0}^N \int_{\Omega_e} \phi_j(x) \,\, \phi_i(x) \, dV \quad \forall \quad \phi_i, \phi_j \in V^e_h \\
        &= \sum_{e=0}^N M_e

    but here we only compute the portion contributed by our current element.
    Hence we only need to consider the trial functions within each elements.
    when integrated in reference element coordinates, :math:`\xi` this is:

    .. math::
        M = \int_0^1 \phi_j(\xi) \phi_i(\xi) det(J)dV

    To evaluate these integras Gaussian quadrature is used such that thi integal becomes:

    .. math::
        M = \sum_{q=0}^{N_q} \phi_j(\xi_q) \phi_i(\xi_q) det(J) w_q

    Parameters
    ----------
    phi_xq : array_like(float), shape((dofe, num_q))
        For each shape function the value at the quadrature points.
    wq_detJ : array_like(float), shape((dofe, num_q))
        Integration weight including the mapping from local to global coordinates.

    Returns
    -------
    me : array_like(float), shape((dofe, dofe))
        Element mass matrix.
    """
    # Create empty storage for element properties.
    dofe = len(phi_xq)
    me = np.zeros((dofe, dofe))

    # Quadrature summations handeled by numpy.sum().
    # Loop over all degrees of freedom and get:
    for i in range(dofe):
        # Matrix diagonal quantiy.
        me[i, i] = np.sum(phi_xq[i] * phi_xq[i] * wq_detJ)

        # Loop over all degrees of freedom and get matrix quantities.
        for j in range(i+1, dofe):
            me_ij = np.sum(phi_xq[j] * phi_xq[i] * wq_detJ)
            me[i, j] = me_ij
            me[j, i] = me_ij
    return me


@nb.jit(nopython=True)
def element_transport(phi_xq, invJ_dphi_xq, wq_detJ):
    r"""
    Compute the elmement transport matrix.

    This matrix is defined as:

    .. math::
        T &= \int_{\Omega} \partial_x\phi_j(x) \,\, \phi_i(x) \,\, dV  \quad \forall \quad \phi_i, \phi_j \in V_h \\
        &= \sum_{e=0}^N \int_{\Omega_e} \partial_x\phi_j(x) \,\, \phi_i(x) \, dV \quad \forall \quad \phi_i, \phi_j \in V^e_h \\
        &= \sum_{e=0}^N T_e

    but here we only compute the portion contributed by our current element.
    Hence we only need to consider the trial functions within each elements.
    when integrated in reference element coordinates, :math:`\xi` this is:

    .. math::
        T_e = \int_0^1 J^{-1}\,\partial_{\xi}\phi_j(\xi) \,\, \phi_i(\xi) \,\, det(J)dV

    To evaluate these integras Gaussian quadrature is used such that thi integal becomes:

    .. math::
        T_e = \sum_{q=0}^{N_q} J^{-1}\,\partial_{\xi}\phi_j(\xi_q) \,\, \phi_i(\xi_q) \,\,det(J) w_q

    Parameters
    ----------
    phi_xq : array_like(float), shape((dofe, num_q))
        For each shape function the value at the quadrature points.
    invJ_dphi_xq : array_like(float), shape((dofs, num_q))
        For each shape function its derivative value at the quadrature points times the inverse Jacobian.
    wq_detJ : array_like(float), shape((dofe, num_q))
        Integration weight including the mapping from local to global coordinates.

    Returns
    -------
    Te : array_like(float), shape((dofe, dofe))
        Element mass matrix.
    """
    # Create empty storage for element properties.
    dofe = len(phi_xq)
    te = np.zeros((dofe, dofe))

    # Quadrature summations handeled by numpy.sum().
    # Loop over all degrees of freedom and get:
    for i in range(dofe):
        # Loop over all degrees of freedom and get matrix quantities.
        for j in range(dofe):
            te[i, j] = np.sum(invJ_dphi_xq[j] * phi_xq[i] * wq_detJ)
    return te


@nb.jit(nopython=True)
def element_stiffness(invJ_dphi_xq, wq_detJ):
    r"""
    Compute the elmement stiffness matrix.

    This matrix is defined as:

    .. math::
        S &= \int_{\Omega} -\partial_x\phi_j(x) \,\, \partial_x\phi_i(x) \,\, dV \quad \forall \quad \phi_i, \phi_j \in V_h \\
        &= \sum_{e=0}^N \int_{\Omega_e} -\partial_x\phi_j(x) \,\, \partial_x\phi_i(x) \,\, dV \quad \forall \quad \phi_i, \phi_j \in V^e_h \\
        &= \sum_{e=0}^N S_e

    but here we only compute the portion contributed by our current element.
    Hence we only need to consider the trial functions within each elements.
    When integrated in reference element coordinates, :math:`\xi` this is:

    .. math::
        S_e = \int_0^1 J^{-1}\, \partial_{\xi} \phi_j(\xi) \,\, J^{-1}\,\partial_{\xi}\phi_i(\xi) \,\,det(J)dV

    To evaluate these integras Gaussian quadrature is used such that thi integal becomes:

    .. math::
        S_e = \sum_{q=0}^{N_q} J^{-1}\, \partial_{\xi} \phi_j(\xi_q) \,\, J^{-1}\,\partial_{\xi}\phi_i(\xi_q) \,\,det(J) w_q

    Parameters
    ----------
    invJ_dphi_xq : array_like(float), shape((dofs, num_q))
        For each shape function its derivative value at the quadrature points times the inverse Jacobian.
    wq_detJ : array_like(float), shape((dofe, num_q))
        Integration weight including the mapping from local to global coordinates.

    Returns
    -------
    Se : array_like(float), shape((dofe, dofe))
        Element mass matrix.
    """
    # Create empty storage for element properties.
    dofe = len(invJ_dphi_xq)
    se = np.zeros((dofe, dofe))

    # Quadrature summations handeled by numpy.sum().
    # Loop over all degrees of freedom and get:
    for i in range(dofe):
        # Matrix diagonal quantiy.
        se[i, i] = -np.sum(invJ_dphi_xq[i] * invJ_dphi_xq[i] * wq_detJ)

        # Loop over all degrees of freedom and get matrix quantities.
        for j in range(i+1, dofe):
            se_ij = -np.sum(invJ_dphi_xq[j] * invJ_dphi_xq[i] * wq_detJ)
            se[i, j] = se_ij
            se[j, i] = se_ij
    return se


@nb.jit(nopython=True)
def element_rhs(phi_xq, wq_detJ, f_xq):
    r"""
    Compute the elmement right hand side vector.

    Parameters
    ----------
    phi_xq : array_like(float), shape((dofs, num_q))
        For each shape function the value at the quadrature points.
    f_xq : array_like(float), shape(num_q)
        The value of the right hand side equation evaluated at the quadrature points.
    wq_detJ : array_like(float), shape((dofs, num_q))
        Integration weight including the mapping from local to global coordinates.

    Returns
    -------
    fe : array_like(float), shape(dofe)
        Element right hand side in our system of equations.
    """
    # Create empty storage for element properties.
    dofe = len(phi_xq)
    fe = np.zeros(dofe)

    # Quadrature summations handeled by numpy.sum().
    # Loop over all degrees of freedom and get:
    for i in range(dofe):
        # Vector quantity.
        fe[i] = np.sum(f_xq * phi_xq[i] *wq_detJ)
    return fe


@nb.jit(nopython=True)
def kernel1d(mesh, rhs=None, mass=False, transport=False, stiffness=False):
    r"""
    Create the global FEM system by looping over the elements.

    Parameters
    ----------
    mesh : Mesh
        The mesh class specifying all discretization.
    rhs : callable
        Function that acts as our right hand side (nonhomogeneous term), set equal to `None` if the rhs is zero valued.
    mass : bool, optional
        Return a mass matrix. Default is `False`.
    transport : bool, optional
        Return the transport matrix. Default is `False`.
    stiffness : bool, optional
        Return the stiffness matrix. Default is `False`.

    Returns
    -------
    f : array_like(float), shape(dofe)
        Global right hand side in our system of equations.
        Only when `rhs != None`, `None` otherwise.
    M : COO (value, (row, column))
        Global mass matrix, ready to be converted to COO. Repeating indices do exist.
        Only `mass == True`, `None` otherwise.
    T : COO (value, (row, column))
        Global transport matrix, ready to be converted to COO. Repeating indices do exist.
        Only 'transport == True`, `None` otherwise.
    S : COO (value, (row, column))
        Global stiffness matrix, ready to be converted to COO. Repeating indices do exist.
        Only 'stiffness == True`, `None` otherwise.
    """
    # num_ele = mlen(c)  # Number of elements.
    # num_dofs = np.max(c) + 1   # Total number of degrees of freedom.
    # num_dofe = len(c[0])  # Number of degrees of freedom per element.

    # Initialize right hand side vector (dense).
    f = np.zeros(mesh.num_dofs, dtype=np.float64)

    # Initialize mass matrix (sparse coo format).
    if mass is True:
        m_v = np.zeros((mesh.num_ele, mesh.num_dofe ** 2), dtype=np.float64)  # Value list
        m_i = np.zeros((mesh.num_ele, mesh.num_dofe ** 2), dtype=np.uintc)  # Row list
        m_j = np.zeros((mesh.num_ele, mesh.num_dofe ** 2), dtype=np.uintc)  # Column list
    else:
        m_v, m_i, m_j = None, None, None

    # Initialize transport matrix (sparse coo format).
    if transport is True:
        t_v = np.zeros((mesh.num_ele, mesh.num_dofe ** 2), dtype=np.float64)  # Value list
        t_i = np.zeros((mesh.num_ele, mesh.num_dofe ** 2), dtype=np.uintc)  # Row list
        t_j = np.zeros((mesh.num_ele, mesh.num_dofe ** 2), dtype=np.uintc)  # Column list
    else:
        t_v, t_i, t_j = None, None, None

    # Initialize stiffness matrix (sparse coo format).
    if stiffness is True:
        s_v = np.zeros((mesh.num_ele, mesh.num_dofe ** 2), dtype=np.float64)  # Value list
        s_i = np.zeros((mesh.num_ele, mesh.num_dofe ** 2), dtype=np.uintc)  # Row list
        s_j = np.zeros((mesh.num_ele, mesh.num_dofe ** 2), dtype=np.uintc)  # Column list
    else:
        s_v, s_i, s_j = None, None, None

    # This is the main loop
    for ele in range(mesh.num_ele):
        # Obtain element properties
        dofe, phi_xq, invJ_dphi_xq, f_xq, wq_detJ = mesh.get_element(ele, rhs=rhs)

        # Perform integration and compute right hand side vector.
        if rhs != None:
            fe = element_rhs(phi_xq, wq_detJ, f_xq)
            f[dofe] += fe

        # Calculate element mass matrix and add to value colunm row arrays.
        if mass == True:
            me = element_mass(phi_xq, wq_detJ)
            ie = np.repeat(dofe, len(dofe))
            je = ie.reshape((-1, len(dofe))).T.ravel()
            m_v[ele] = me.ravel()
            m_i[ele] = ie
            m_j[ele] = je

        if transport == True:
            te = element_transport(phi_xq, invJ_dphi_xq, wq_detJ)
            ie = np.repeat(dofe, len(dofe))
            je = ie.reshape((-1, len(dofe))).T.ravel()
            t_v[ele] = te.ravel()
            t_i[ele] = ie
            t_j[ele] = je

        if stiffness == True:
            se = element_stiffness(invJ_dphi_xq, wq_detJ)
            ie = np.repeat(dofe, len(dofe))
            je = ie.reshape((-1, len(dofe))).T.ravel()
            s_v[ele] = se.ravel()
            s_i[ele] = ie
            s_j[ele] = je

    # Flatten mass COO arrays, repeating indices remain.
    if mass == True:
        M = (m_v.ravel(), (m_i.ravel(), m_j.ravel()))
    else:
        M = None

    if transport == True:
        T = (t_v.ravel(), (t_i.ravel(), t_j.ravel()))
    else:
        T = None

    if stiffness == True:
        S = (s_v.ravel(), (s_i.ravel(), s_j.ravel()))
    else:
        S = None

    return M, T, S, f


@nb.jit(nopython=True)
def interpolate(mesh, u, x_inter):
    r"""
    Obtain the field :math:`u(x)` any points `x_inter` following the FE interpolation.

    Parameters
    ----------
    mesh : Mesh
        The mesh class specifying all discretization.
    u : array_like(float), shape(dofs)
        The field `u` at the degrees of freedom.
    x_inter : array_like(float)
        The location where we want to obtain our interpolated field.

    Returns
    -------
    array_like(float)
        The field `u` at the interpolation points `x_inter`.
    """
    # Initialize storage for our interpolated field.
    u_inter = np.zeros_like(x_inter, dtype=np.float64)

    # Loop over all elements and interpolate the inside.
    for ele in range(mesh.num_ele):
        # Obtain element properties
        dofe = mesh.connectivity[ele]
        x_ele = mesh.nodes[ele]

        # Find which values of x are within our element.
        ind = np.where((x_ele[0] <= x_inter) & (x_inter <= x_ele[1]))

        # Obtain the element coordinates and calculate the field.
        xi = mesh.x_to_xi(ele, x_inter[ind])
        u_inter[ind] = np.sum(u[dofe] * mesh.shape(xi).T, axis=1)

    return u_inter
