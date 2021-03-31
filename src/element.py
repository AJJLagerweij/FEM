r"""
Discretization objects, containing both the meshing and the solution space approximation.

That is inside this object is both the :math:`h` and :math:`p` discretization.
These are in this code orginized together as no local :math:`p` refinement is expeced.
There is a base class, Mesh, specifying the interface to the main kernel and solver,
and there are the following interited classes specifying:

1. Mesh1D for a 1D mesh of different approximation orders :math:`p`.
2. Mesh2Dtri for a 2D mesh of triangles.
3. Mesh2Dqua for a 2D mesh of quadralatirals.

And although these have the name 'mesh' they do describe the elements as well.

Bram Lagerweij
COHMAS Mechanical Engineering KAUST
2021
"""

# Importing required modules.
import numba as nb
import numpy as np

# Import my own scripts.
from helper import gauss, quadtri


# Defince type specification of the class attributies.
spec = [('num_ele', nb.uint),
        ('order', nb.uint),
        ('num_dofe', nb.uint),
        ('num_dofs', nb.uint),
        ('num_q', nb.uint),
        ('nodes', nb.float64[:, :]),
        ('connectivity', nb.uint[:, :])]


class Mesh(object):
    r"""
    Specify a base mesh object, and it's behaviour.

    This base class is not an actual usefull mesh but provides the
    basics outline that should be in all other mesh classes.
    All other meshes should be childeren from this base class.
    But inheritance works badly with the just in time compilation of `numba`.
    Hence all function have to be redefined in the child classes, while the class
    requires the `@nb.jitclass(spec)` decorator.

    Attributes
    ----------
    num_ele : int
        Number of elements in the entire mesh.
    num_dofe : int
        Number of degrees of freedom per element.
    num_dofs : int
        Number of degrees of freedom in the problem, this property
        depends on the element type and the mesh.
    num_q : int
        Number of quadrature point in integration approximations.
    nodes : array_like(float), shape(n+1, order+1)
        For each node in each element the coordinates.
    connectivity : array_like(int), shape(n+1, order+1)
        Elements to node connectivity array.
    """
    def __init__(self):
        r"""
        Create dummy mesh, all atributes are `None`.
        """
        # Set attributes.
        self.num_ele = None
        self.num_q = None

        # Obtain properties that define mesh and elements.
        # Create actual mesh.
        self.nodes = None
        self.connectivity = None
        self.num_dofs = None

        # Initialize element properties.
        self.num_dofe = None

    def get_element(self, ele, rhs=None):
        r"""
        Get the element properties of element `ele`.

        Parameters
        ----------
        ele : int
            Number of the element of which the properties should be obtained.
        rhs : callable(float)
            The righthandside function of the pde in terms of global coordinates.

        Returns
        -------
        dofe : array_like(int), shape(num_dofe)
            The degrees of freedom that belong to this element.
        phi_xq : array_like(float), shape((dofs, num_q))
            For each shape function the value at the quadrature points.
        invJ_dphi_xq : array_like(float), shape((dofs, num_q))
            For each shape function its derivative value at the quadrature points times the inverse Jacobian.
        f_xq : array_like(float), shape(num_q)
            The value of the right hand side equation evaluated at the quadrature points.
        wq_detJ : array_like(float), shape((dofs, num_q))
            For the local determinant times quadrature weight at each of the quadrature points.
        """
        raise NotImplementedError("This base class is empty.")
        return dofe, phi_xq, invJ_dphi_xq, f_xq, wq_detJ

    def x_to_xi(self, ele, x):
        r"""
        Converting global into local coordinates :math:`x \rightarrow \xi`.

        Parameters
        ----------
        ele : int
            Element in which the transformation has to take place.
        x : array_like(float)
            Global coordinates, these must be within the element.

        Returns
        -------
        xi : array_like(float)
            The local, element, coordinates.
        """
        raise NotImplementedError("This base class is empty.")
        return xi

    def xi_to_x(self, ele, xi):
        r"""
        Converting local coordinates into global ones :math:`\xi \rightarrow x`.

        Parameters
        ----------
        ele : int
            Element in which the transformation has to take place.
        xi : array_like(float)
            Local coordinates within the element.

        Returns
        -------
        x : array_like(float)
            The global coordinates.
        """
        raise NotImplementedError("This base class is empty.")
        return x

    def jacobian(self, ele, xi):
        r"""
        The jacobian and mapping for the local to global coordinates system (:math:`\xi` to :math:`x`).

        Parameters
        ----------
        ele : int
            Element for which the jacobian has to be calculated.
        xi : array_like(float)
            Location where the jacobians should be measured.

        Returns
        -------
            jac : array_like(float)
                The Jacobian at :math:`\xi`.
            invJ : array_like(float)
                The inverse Jacobian at :math:`\xi`.
            detJ : array_like(float)
                The derivative of the Jacobian at :math:`\xi`.
        """
        raise NotImplementedError("This base class is empty.")
        return jac, invJ, detJ

    def shape(self, xi):
        r"""
        Shape functions at locations :math:`\xi` in element coordinates system.

        Parameters
        ----------
        xi : array_like(float)
            Locations :math:`\xi` where the shape functions are evaluated.

        Returns
        -------
        phi_xq : array_like(float)
            Shape functions :math:`\phi_i` at locations :math:`\xi`.
        """
        raise NotImplementedError("This base class is empty.")
        return phi_xq

    def dshape(self, xi):
        r"""
        Shape functions derivatives at locations :math:`\xi` in element coordinates system.

        Parameters
        ----------
        xi : array_like(float)
            Locations :math:`\xi` where the shape functions are evaluated.

        Returns
        -------
        dphi_xq : array_like(float)
            Shape functions derivatives :math:`\phi_i` at locations :math:`\xi`.
        """
        raise NotImplementedError("This base class is empty.")
        return dphi_xq


@nb.experimental.jitclass(spec)
class Mesh1D(Mesh):
    r"""
    Specify a 1D mesh object, and it's behaviour.

    This is a 1D mesh object with Lagransian basis functions.

    Parameters
    ----------
    x_start : float
        Start coordinate of the domain.
    x_end : float
        End coordinate of the domain.
    num_ele : int
        Number of elements in the mesh.
    order : int
        Polynomial order of the Lagransian basis functions.
    num_q : int
        Number of quadrature points per element.
    periodic : bool, optional
        Whether the domain is periodic, default is `False`.

    Attributes
    ----------
    num_ele : int
        Number of elements in the entire mesh.
    order : int
        Order of the polynomaial approximation.
    num_dofe : int
        Number of degrees of freedom per element.
    num_dofs : int
        Number of degrees of freedom in the problem, this property
        depends on the element type and the mesh.
    num_q : int
        Number of quadrature point in integration approximations.
    nodes : array_like(float), shape(n+1, order+1)
        For each node in each element the coordinates.
    connectivity : array_like(int), shape(n+1, order+1)
        Elements to node connectivity array.
    """

    def __init__(self, x_start, x_end, num_ele, order, num_q, periodic=False):
        # Determine basic nodal properties.
        num_dofe = order + 1
        num_dofs = num_ele * order + 1

        # Create mesh by discributing nodes and creating connectivity.
        nodes_x = np.linspace(x_start, x_end, num_ele + 1).astype(np.float64)
        nodes = np.zeros((num_ele, 2), dtype=np.float64)
        connectivity = np.zeros((num_ele, num_dofe), dtype=np.uint)
        for i in range(num_ele):
            nodes[i][0] = nodes_x[i]
            nodes[i][1] = nodes_x[i + 1]
            for j in range(num_dofe):
                connectivity[i][j] = nb.uint(i * (num_dofe - 1) + j)

        # Make mesh periodic if this is required.
        if periodic is True:
            connectivity[-1, -1] = 0

        # Create actual mesh.
        self.num_ele = num_ele
        self.nodes = nodes  # nodes
        self.connectivity = connectivity  # connectivity
        self.num_dofs = int(connectivity.max() + 1)

        # Initialize element properties.
        self.order = order
        self.num_dofe = num_dofe
        self.num_q = num_q

    def get_element(self, ele, rhs=None):
        r"""
        Get the element properties of element `ele`.

        Parameters
        ----------
        ele : int
            Number of the element of which the properties should be obtained.
        rhs : callable(float), optional
            The righthandside function of the pde in terms of global coordinates.

        Returns
        -------
        dofe : array_like(int), shape(num_dofe)
            The degrees of freedom that belong to this element.
        phi_xq : array_like(float), shape((dofs, num_q))
            For each shape function the value at the quadrature points.
        invJ_dphi_xq : array_like(float), shape((dofs, num_q))
            For each shape function its derivative value at the quadrature points times the inverse Jacobian.
        f_xq : array_like(float), shape(num_q)
            The value of the right hand side equation evaluated at the quadrature points.
        wq_detJ : array_like(float), shape((dofs, num_q))
            For the local determinant times quadrature weight at each of the quadrature points.
        """
        # Obtain the global node numbers of this element.
        dofe = self.connectivity[ele]

        # Get the Quadrature weights.
        xq, wq = gauss_tri(self.num_q)

        # Get information on element coordinate transformation.
        jac, invJ, detJ = self.jacobian(ele)

        # Get properties at quadrature points.
        phi_xq = self.shape(xq)
        dphi_xq = self.dshape(xq)
        invJ_dphi_xq = invJ * dphi_xq
        wq_detJ = wq * detJ

        # Check if the right hand side is a function or not.
        if rhs != None:
            x_xq = self.xi_to_x(ele, xq)
            f_xq = rhs(x_xq)
        else:
            f_xq = np.zeros_like(xq)
        return dofe, phi_xq, invJ_dphi_xq, f_xq, wq_detJ

    def x_to_xi(self, ele, x):
        r"""
        Converting local coordinates into global ones :math:`x \rightarrow \xi`.

        Parameters
        ----------
        ele : int
            Element in which the transformation has to take place.
        x : array_like(float)
            Global coordinates, these must be within the element.

        Returns
        -------
        xi : array_like(float)
            The local, element, coordinates.
        """
        jac_inv = self.jacobian(ele)[1]
        x0 = self.nodes[ele][0]
        xi = jac_inv * (x - x0)
        return xi

    def xi_to_x(self, ele, xi):
        r"""
        Converting local coordinates into global ones :math:`\xi \rightarrow x`.

        Parameters
        ----------
        ele : int
            Element in which the transformation has to take place.
        xi : array_like(float)
            Local coordinates within the element.

        Returns
        -------
        x : array_like(float)
            The global coordinates.
        """
        jac = self.jacobian(ele)[0]
        x0 = self.nodes[ele][0]
        x = jac * xi + x0
        return x

    def jacobian(self, ele):
        r"""
        The jacobian and mapping for the local to global coordinates system (:math:`\xi` to :math:`x`).

        Because the jacobian is a constant for 1D meshes, the objectes that are returned are constant
        floats instead of arrays.

        Parameters
        ----------
        ele : int
            Element for which the jacobian has to be calculated.

        Returns
        -------
            jac : float
                The Jacobian at :math:`\xi`.
            invJ : float
                The inverse Jacobian at :math:`\xi`.
            detJ : float
                The derivative of the Jacobian at :math:`\xi`.
        """
        x = self.nodes[ele]
        jac = x[1] - x[0]
        detJ = jac
        invJ = 1 / jac
        return jac, invJ, detJ

    def shape(self, xi):
        r"""
        Shape functions at locations :math:`\xi` in element coordinates system.

        Parameters
        ----------
        xi : array_like(float)
            Locations :math:`\xi` where the shape functions are evaluated.

        Returns
        -------
        phi_xq : array_like(float)
            Shape functions :math:`\phi_i` at locations :math:`\xi`.
        """
        phi = np.zeros((self.order + 1, len(xi)))

        if self.order == 1:
            phi[0] = 1 - xi
            phi[1] = xi
        elif self.order == 2:
            phi[0] = 1 - 3*xi + 2*xi**2
            phi[2] = -xi + 2*xi**2
            phi[1] = 4*xi - 4*xi**2
        else:
            raise NotImplementedError("This order of shape function is not implemented.")

        return phi

    def dshape(self, xi):
        r"""
        Shape functions derivatives at locations :math:`\xi` in element coordinates system.

        Parameters
        ----------
        xi : array_like(float)
            Locations :math:`\xi` where the shape functions are evaluated.

        Returns
        -------
        dphi_xq : array_like(float)
            Shape functions derivatives :math:`\phi_i` at locations :math:`\xi`.
        """
        dphi = np.zeros((self.order + 1, len(xi)))

        if self.order == 1:
            dphi[0] = -1
            dphi[1] = 1
        elif self.order == 2:
            dphi[0] = -3 + 4 * xi
            dphi[2] = -1 + 4 * xi
            dphi[1] = 4 - 8 * xi
        else:
            raise NotImplementedError("This order of shape function is not implemented.")

        return dphi


@nb.experimental.jitclass(spec)
class Mesh2Dtri(Mesh):
    r"""
    Specify a 2D mesh of triagular elements, and it's behaviour.

    This is a 1D mesh object with Lagransian basis functions.

    Parameters
    ----------
    x_start : float
        Start coordinate of the domain.
    x_end : float
        End coordinate of the domain.
    num_ele : int
        Number of elements in the mesh.
    order : int
        Polynomial order of the Lagransian basis functions.
    num_q : int
        Number of quadrature points per element.
    periodic : bool, optional
        Whether the domain is periodic, default is `False`.

    Attributes
    ----------
    num_ele : int
        Number of elements in the entire mesh.
    order : int
        Order of the polynomaial approximation.
    num_dofe : int
        Number of degrees of freedom per element.
    num_dofs : int
        Number of degrees of freedom in the problem, this property
        depends on the element type and the mesh.
    num_q : int
        Number of quadrature point in integration approximations.
    nodes : array_like(float), shape(n+1, order+1)
        For each node in each element the coordinates.
    connectivity : array_like(int), shape(n+1, order+1)
        Elements to node connectivity array.
    """

    def __init__(self, deva, num_q):
        # Determine basic nodal properties.


        # Make mesh periodic if this is required.
        if periodic is True:
            connectivity[-1, -1] = 0

        # Create actual mesh.
        self.num_ele = num_ele
        self.nodes = nodes  # nodes
        self.connectivity = connectivity  # connectivity
        self.num_dofs = int(connectivity.max() + 1)

        # Initialize element properties.
        self.order = order
        self.num_dofe = num_dofe
        self.num_q = num_q

    def get_element(self, ele, rhs=None):
        r"""
        Get the element properties of element `ele`.

        Parameters
        ----------
        ele : int
            Number of the element of which the properties should be obtained.
        rhs : callable(float), optional
            The righthandside function of the pde in terms of global coordinates.

        Returns
        -------
        dofe : array_like(int), shape(num_dofe)
            The degrees of freedom that belong to this element.
        phi_xq : array_like(float), shape((dofs, num_q))
            For each shape function the value at the quadrature points.
        invJ_dphi_xq : array_like(float), shape((dofs, num_q))
            For each shape function its derivative value at the quadrature points times the inverse Jacobian.
        f_xq : array_like(float), shape(num_q)
            The value of the right hand side equation evaluated at the quadrature points.
        wq_detJ : array_like(float), shape((dofs, num_q))
            For the local determinant times quadrature weight at each of the quadrature points.
        """
        # Obtain the global node numbers of this element.
        dofe = self.connectivity[ele]

        # Get the Quadrature weights.
        xq, wq = gauss(self.num_q)

        # Get information on element coordinate transformation.
        jac, invJ, detJ = self.jacobian(ele)

        # Get properties at quadrature points.
        phi_xq = self.shape(xq)
        dphi_xq = self.dshape(xq)
        invJ_dphi_xq = invJ * dphi_xq
        wq_detJ = wq * detJ

        # Check if the right hand side is a function or not.
        if rhs != None:
            x_xq = self.xi_to_x(ele, xq)
            f_xq = rhs(x_xq)
        else:
            f_xq = np.zeros_like(xq)
        return dofe, phi_xq, invJ_dphi_xq, f_xq, wq_detJ

    def x_to_xi(self, ele, x):
        r"""
        Converting local coordinates into global ones :math:`x \rightarrow \xi`.

        Parameters
        ----------
        ele : int
            Element in which the transformation has to take place.
        x : array_like(float)
            Global coordinates, these must be within the element.

        Returns
        -------
        xi : array_like(float)
            The local, element, coordinates.
        """

        return xi

    def xi_to_x(self, ele, xi):
        r"""
        Converting local coordinates into global ones :math:`\xi \rightarrow x`.

        Parameters
        ----------
        ele : int
            Element in which the transformation has to take place.
        xi : array_like(float)
            Local coordinates within the element.

        Returns
        -------
        x : array_like(float)
            The global coordinates.
        """

        return x

    def jacobian(self, ele):
        r"""
        The jacobian and mapping for the local to global coordinates system (:math:`\xi` to :math:`x`).

        Because the jacobian is a constant for 1D meshes, the objectes that are returned are constant
        floats instead of arrays.

        Parameters
        ----------
        ele : int
            Element for which the jacobian has to be calculated.

        Returns
        -------
            jac : float
                The Jacobian at :math:`\xi`.
            invJ : float
                The inverse Jacobian at :math:`\xi`.
            detJ : float
                The derivative of the Jacobian at :math:`\xi`.
        """
        x = self.nodes[ele]

        return jac, invJ, detJ

    def shape(self, xi):
        r"""
        Shape functions at locations :math:`\xi` in element coordinates system.

        Parameters
        ----------
        xi : array_like(float)
            Locations :math:`\xi` where the shape functions are evaluated.

        Returns
        -------
        phi_xq : array_like(float)
            Shape functions :math:`\phi_i` at locations :math:`\xi`.
        """
        phi = np.zeros((self.order + 1, len(xi)))



        return phi

    def dshape(self, xi):
        r"""
        Shape functions derivatives at locations :math:`\xi` in element coordinates system.

        Parameters
        ----------
        xi : array_like(float)
            Locations :math:`\xi` where the shape functions are evaluated.

        Returns
        -------
        dphi_xq : array_like(float)
            Shape functions derivatives :math:`\phi_i` at locations :math:`\xi`.
        """
        dphi = np.zeros((self.order + 1, len(xi)))



        return dphi


# def triangle2d(x, order):
#     r"""
#     Shape function for a 2D linear triangles.
#
#     Parameters
#     ----------
#     x : array_like (float, float)
#         Location [x1, x2] where the shape functions are sampled.
#     order : int
#         The order of the polynomial used, only option is 1.
#
#     Returns
#     -------
#     array_like
#         The array with the values of the shape funcions at `x`.
#
#     Raises
#     ------
#     NotImplementedError
#         Raised when the requested order of shape function polynomaial is not implemented.
#     """
#     phi = np.zeros((order+1, len(x)))
#     dxphi = np.zeros_like(phi)
#     dyphi = np.zeros_like(phi)
#
#     if order == 1:
#         # Shape functions
#         phi[0] = x[0]
#         phi[1] = x[1]
#         phi[2] = 1 - x[0] - x[1]
#
#         # Derivatives
#         dxphi[0] = 1
#         dxphi[1] = 0
#         dxphi[2] = -1
#         dyhpi[0] = 0
#         dyhpi[0] = 1
#         dyhpi[0] = -1
#
#     else:
#         raise NotImplementedError("This order of shape function is not implemented.")
#
#     return phi, dphi
