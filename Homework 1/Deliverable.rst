.. |br| raw:: html

   <br />

##########
Homework 1
##########

.. admonition:: Toppic

   Homework regarding the first week. The goal is  to  work with basic numerical approximation of PDE's' and functions.

   Bram Lagerweij |br|
   08 Feb 2020



*****************
1 Method of Lines
*****************
Consider the one-dimensional advection diffusion equation:

.. math::
	{u}_{t} + {u}_{x} + \mu{u}_{xx} = 0 \qquad \forall \, x \in \Omega = [0, 1] \quad \& \quad t>0

where :math:`mu>0` is a coefficient.  Consider periodic boundary conditions and the following initial condition:

.. math::
	u(x,0) = \sin(2\pi x)

What do we expect the exact solution to do? Due to the advective part, the initial condition travels at constant speed to the right.
At the same time, due to the diffusive term, the initial condition is dissipated at a rate that depends on :math:`\mu`.

Consider the following discretization. Use second-order central finite differences to approximate :math:`u_x` and :math:`u_{xx}`.
Use forward and backward Euler to obtain full discretization (write down the schemes). Consider a fixed mesh with of :math:`\Delta x`.

1.1 Advective Diffusive PDE
===========================
Consider a final time of :math:`t=1` and :math:`\mu=0.01`. For each full discretization proceed as follows:

1. Experiment using the following time step sizes: :math:`\Delta t = 10^{−4},\, 10^{−3}` and :math:`10^{−1}`. 
2. How do the explicit and implicit methods behave for these time steps?

1.2 Advective PDE
=================
Consider :math:`\mu=0` and solve the PDEusing the explicit and the implicit methods.
Use :math:`\Delta t = 10^{−4}` and solve the problem for the following final times :math:`t=1,\, 5,\, 10,\, 15` and :math:`20`.
Comment on the behaviour of each full discretization as the final time increases.


****************************
2 Approximation of functions
****************************
Consider the function:

.. math::
    f(x) = \sin^4(2\pi x) \qquad \forall \, x \in \Omega = [0, 1]

for which we have to find multiple global and local approximations.
Let :math:`f_h (x)` be such an approximation for a given grid. We consider the following errors:

.. math::
    E_1 := \int_\Omega | f(x) - f_h(x) | dx \quad \text{and} \quad E_2 := \int_\Omega \big(f(x) - f_h(x)\big)^2 dx

2.1 Global Approximations
=========================
Consider the following approximations all with :math:`N` terms:

1. the Taylor series around :math:`x=0.5`,
2. the Fourier series,
3. a global polynomial interpolation on the closed interval given by:
 
.. math::
    f_h(x) = a_0 + a_1 x + a_2 x^2 + \dots + a_N x^N

Consider different levels of refinement, :math:`N=4,\, 5,\, 6,\,\dots,\,10` and for each approximation report both :math:`E_1` and :math:`E_2`.

2.2 Local Approximations
========================
Split the domain :math:`\Omega` into :math:`N` cells. For each cell :math:`K`, compute linear and quadratic approximations :math:`f_K(x)` where :math:`f_K(x_i)=f(x_i)` where :math:`x_i` are evenly spaced gridpoints, including the boundaries of the cell.
Comput and report both :math:`E_1` and :math:`E_2` for a different numbers of cells :math:`N=4,\, 5,\, 6,\,\dots,\,10`.