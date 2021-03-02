.. |br| raw:: html

   <br />

.. math::
    \require{physics}
    \renewcommand{\vec}[1]{\underline{#1}}
    \renewcommand{\exp}{\text{e}}
    \DeclareMathOperator*{\argmin}{argmin}

##########
Homework 2
##########

.. admonition:: Topic

   Homework regarding the third week. The goal is to work with simple 1D FEM methods.
   We'll be solving several PDE's and project function on a FEM space.

   Bram Lagerweij |br|
   2 Mar 2021

.. contents:: Table of Contents
    :local:
    :depth: 2

*************************************
1 Project the Navier-Stokes equations
*************************************
Consider the incompressible Navier-Stokes equations in non-conservative form:

.. math::
   \partial_t \vec{u} + \grad{\vec{u}}\, \vdot \vec{u} +\frac{1}{\rho} \grad{p} - \mu \grad^2{\vec{u}} = \vec{f} \qquad &\forall \quad \vec{x}\in\Omega\\
   \divergence{\vec{u}} = 0 \qquad &\forall \quad \vec{x}\in\Omega\\
   \vec{u}\vdot\vec{n} = 0 \qquad &\forall\quad \vec{x}\in\Omega

where :math:`\vec{u}, \vec{x}, \vec{f}, \vec{n}\in\mathbb{R}^d` are the speed, location, external forces and surface normal, :math:`\rho` the density, :math:`\mu` the viscosity and :math:`p` the pressure.
The original Chorin's projection method considers the following discretziation in time:

.. math::
   \frac{\vec{u}^*-\vec{u}^n}{\Delta t} + \grad{\vec{u}^n} \, \vdot \vec{u}^n - \mu \grad^2{\vec{u}^*} = \vec{f}

where we ingore the pressure as a kind of operation splitting.
The non-linear term is treated explicitely to avoid the non-linearity and we treat the viscouse term implicitely to avoid extreme small time step restrictions.
However this does not ensure that :math:`\divergence{\vec{u}^*}=0`.
To fix this, the projection method considers:

.. math::
   \frac{\vec{u}^{n+1} - \vec{u}^*}{\Delta t} = -\frac{1}{\rho} \grad p^{n+1}

When we take the divergence we impose :math:`\divergence{\vec{u}^{n+1}}=0` to get:

.. math::
   \Delta p^{n+1} = \frac{\rho}{\Delta t} \divergence{\vec{u}^*}

Finaly, the updated divergence-free velocity is given by:

.. math::
   \vec{u}^{n+1} = \vec{u}^* - \frac{\Delta t}{\rho}\grad p^{n+1}

1.1 The shape functions
=======================
Consider two discrete spaces. For the velocity and pressure use continuous piecewise bi-quadratic and bilinear polynomials (in 2D)

.. math::
   & p_1(x,y) = c_0 x + c_1 y + c_2 xy + c3\\
   & p_2(x,y) = c_0 x^2 + c_1 x^2y + c_2 x^2y^2 + c_3 y^2 + c_4 xy^2 + c_5 x + c_6 y + c_7 xy + c_8

respectively.
How many shape function do we have for each space in the reference element?
Derive the shape functions for the reference element (hint: use tensor products).
Plot these shape functions for both spaces.

1.2 Weak form of Chorin's projection
====================================
Consider the previously described NS-equations and the Chorin'n projection method and obtain:

1. weak formulation
2. discreate weak form
3. the linear algabra representation of the problem.

***************************************************
2 Project a smooth function to :math:`C^0` FE space
***************************************************
From HW1 we consider the following function again:

.. math::
   f(x) = \sin^4(2\pi x) \quad \forall \quad 0 \leq x \leq 1

and project it on the finite element space.

2.1 Projection
==============
Perform the projection through the following steps.
1. Consider piecewise linear and quadratic continuous polynomials.
2. Consider the reference element :math:`[0, 1]` and interpolatory basis functions to derive the shape functions for each space.
3. What is the weak formulation and the linear algebra problem associated with the projection?
4. Compute the entries of the mass matrix for each space.
5. Solve the system to obtain the DoF associated with the projection.
6. Plot the projected functions considering :math:`N = 25, 50, 100` and :math:`200` cells.

.. _2.2 Projection:

2.2 Evaluate Projection
=======================
For both projections compute the followin two errors

.. math::
   E_1 = \int_0^1 \| f(x) - f_h(x) \| \dd{x} \qand E_2 = \sqrt(\int_0^1 (f(x)- f_h(x))^2 \dd{x})

where :math:`f_h(x)` is the projection of :math:`f(x)` on our FE space.
Estimate the order of convergence for each space.
That is assume that the error behaves as:

.. math::
   E = c h^p

where :math:`c` is a constant and :math:`h=1/N` is the mesh size. Whan is the value of :math:`p`?
Does this error behave different for the different spaces and norms?

*******************************************************
3 Project a non-smooth function to :math:`C^0` FE space
*******************************************************
Preform the same projection for the following non-smooth function:

.. math::
   f(x) = \begin{cases} 1 & 0.35 \leq x \leq 0.65 \\ 0 & \text{otherwise} \end{cases}

***************************************
4 Solve Advection-Diffusion PDE with FE
***************************************
Consider the one-dimensional advection diffusion equation:

.. math::
   u_t + u_x - \mu u_{xx} =0 \qquad \forall \quad x\in\Omega=[0,1]

where :math:`\mu>0` is a coefficient.
Consider periodic boundary conditions and the following initial conditions:

.. math::
   u(x, 0) = \sin^4 (2\pi x)

The exact solution to this equation is given by:

.. math::
   u(x,t) = \frac{3}{8} - \frac{1}{2} \exp^{-4\mu t} \cos(2(x-t)) + \frac{1}{8} \exp^{-16\mu t}\cos(4(x-t))

4.1 Solve through FEM
=====================
Solve this problem using a FEM implementation with the following steps:
1. Consider continuous piecewise linear polynomials and interpolatory basis functions.
2. Obtain the discrete weak formulation.
3. Identify the different matrices associated with the finite element discretization.
4. Implement and solve the equation via finite elements up to :math:`t = 1`.

4.2 Compute the error
=====================
Compute the errors :math:`E_1` and :math:`E_2` and compare the results to those of previous weeks homework, in which the same PDE was solved using a Finite Difference approach.
Preforme a convergence test as described in :ref:`2.2 Projection`.