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
   We'll be solving several PDEs and project function on a FEM space.

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
   & p_1(x,y) = c_0 x + c_1 y + c_2 xy + c_3\\
   & p_2(x,y) = c_0 x^2 + c_1 x^2y + c_2 x^2y^2 + c_3 y^2 + c_4 xy^2 + c_5 x + c_6 y + c_7 xy + c_8

respectively.
How many shape function do we have for each space in the reference element?
Derive the shape functions for the reference element (hint: use tensor products).
The code used to plot these two figures is available in `1 shape2D.py <https://github.com/AJJLagerweij/FEM/blob/main/Homework-2/1_shape2D.py>`_.

.. figure:: ../../Homework-2/images/Linear_Quads.svg
    :name: Linear_Quads
    :align: center
    :width: 600

    : Quadrilateral elements with linear shape functions.

.. figure:: ../../Homework-2/images/Quadratic_Quads.svg
    :name: Quadratic_Quads
    :align: center
    :width: 600

    : Quadrilateral elements with quadratic shape functions.

1.2 Weak form of Chorin's projection
====================================
Consider the previously described NS-equations and the Chorin'n projection method and obtain:

1. Weak formulation,
    From what I understand the this approach goes in three steps:

    1. Solve PDE 1,
        .. math::
            \frac{\vec{u}^*-\vec{u}^n}{\Delta t} + \grad{\vec{u}^n} \,\vdot\, \vec{u}^n - \mu \grad^2{\vec{u}^*} = \vec{f}
    
        where :math:`\vec{u}^*` is the unknown and all other variables are known.
        This is PDE can be written into the following format:

        .. math::
            \alpha \vec{u}^* - \beta \grad^2{\vec{u}^*} = \vec{b}_1

        which is a non-homogeneous diffusion equation, but vector valued, as :math:`\vec{u}` is a vector.
    2. Solve PDE 2:
        .. math::
            \grad^2 {p^{n+1}} = \frac{\rho}{\Delta t} \divergence{\,\vec{u}^*}
    
        where :math:`p` is the variable to be determined, through the a Poisson equation.

        .. math::
            \grad^2 {p^{n+1}} = \vec{b}_2

    3. Obtain new primal :math:`\vec{u}` by updating it through:
        .. math::
            \vec{u}^{n+1} = \vec{u}^* - \frac{\Delta t}{\rho}\grad p^{n+1}
        
        This is simply an update, there is no PDE to be solved and everything on the right hand side is known.
    
2. Discrete weak form and,
    Incomplete.
3. The linear algebra representation of the problem.
    I'll first need to find the answer to the previous question, nevertheless it is clear that we need at
    least the mass and the stiffness matrix for the first PDE. The transport matrix is also
    required to compute the right hand side in the first PDE.

***************************************
2 Project a smooth function to FE space
***************************************
From HW1 we consider the following function again:

.. math::
   f(x) = \sin^4(2\pi x) \quad \forall \quad 0 \leq x \leq 1

and project it on the finite element space.

2.1 Projection
==============
Perform the projection through the following steps.

1. Consider piecewise linear and quadratic continuous polynomials.
    Done, see :func:`element.shape1d`.
2. Consider the reference element :math:`[0, 1]` and interpolatory basis functions to derive the shape functions for each space.
    Done, see :func:`element.shape1d`.
3. What is the weak formulation and the linear algebra problem associated with the projection?
    The derivation of the weak form is described at :func:`pde.projection`.
4. Compute the entries of the mass matrix for each space.
    Done, see :func:`fem.element_mass`.
5. Solve the system to obtain the DoF associated with the projection.
    Done, see :func:`solvers.solve`.
6. Plot the projected functions considering :math:`N = 25, 50, 100` and :math:`200` cells.
    Done, the main code, in `2 ProjectionFE.py <https://github.com/AJJLagerweij/FEM/blob/main/Homework-2/2_Projection_FE.py>`_, was used to create :numref:`Smooth_Linear_Elements` and :numref:`Smooth_Quadratic_Elements`.

.. figure:: ../../Homework-2/images/Smooth_Linear_Elements.svg
    :name: Smooth_Linear_Elements
    :align: center
    :width: 600

    : Approximating :math:`f(x)` with a finite element projections with :math:`N` linear elements.

.. figure:: ../../Homework-2/images/Smooth_Quadratic_Elements.svg
    :name: Smooth_Quadratic_Elements
    :align: center
    :width: 600

    : Approximating :math:`f(x)` with a finite element projections with :math:`N` quadratic elements.

.. _2.2 Projection:

2.2 Evaluate Projection
=======================
For both projections compute the following two errors

.. math::
   E_1 = \int_0^1 \| f(x) - f_h(x) \| \dd{x} \qand E_2 = \sqrt{\int_0^1 (f(x)- f_h(x))^2 \dd{x}}

where :math:`f_h(x)` is the projection of :math:`f(x)` on our FE space.

.. figure:: ../../Homework-2/images/Smooth_E1_vs_Elements.svg
    :name: Smooth_E1_vs_Elements
    :align: center
    :width: 600

    : Comparing error 1 to the number of elements shows faster convergence of the quadratic elements.
    The order seems to be 2 and 3 respectively.

.. figure:: ../../Homework-2/images/Smooth_E2_vs_Elements.svg
    :name: Smooth_E2_vs_Elements
    :align: center
    :width: 600

    : Comparing error 2 to the number of elements shows faster convergence of the quadratic elements.
    The order seems to be 2 and 3 respectively.

.. figure:: ../../Homework-2/images/Smooth_E1_vs_DOFs.svg
    :name: Smooth_E1_vs_DOFs
    :align: center
    :width: 600

    : Comparing error 1 to the amount of degrees of freedom still shows faster convergence of the quadratic elements.
    Clearly the difference is less pronounced, because the quadratic elements have more unknowns per element.
    The order seems to be 2 and 3 respectively.

.. figure:: ../../Homework-2/images/Smooth_E2_vs_DOFs.svg
    :name: Smooth_E2_vs_DOFs
    :align: center
    :width: 600

    : The result for error 2 is again similar to that for error 1.
    The order seems to be 2 and 3 respectively.

Estimate the order of convergence for each space.
That is assume that the error behaves as:

.. math::
   E = c h^p

where :math:`c` is a constant and :math:`h=1/N` is the mesh size. When is the value of :math:`p`?
Does this error behave different for the different spaces and norms?

+--------+-----------------------------------------------------+-----------------------------------------------------+
|        |                        Linear                       |                      Quadratic                      |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| N      | E1 p(1/N) | E2 p(1/N) | E1 p(1/DoFs) | E2 p(1/DoFs) | E1 p(1/N) | E2 p(1/N) | E1 p(1/DoFs) | E2 p(1/DoFs) |
+========+===========+===========+==============+==============+===========+===========+==============+==============+
| 4      | 1.93      | 1.90      | 2.61         | 2.57         | 1.98      | 1.93      | 2.33         | 2.28         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 8      | 1.17      | 1.16      | 1.38         | 1.37         | 2.09      | 2.02      | 2.28         | 2.20         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 16     | 1.83      | 1.67      | 1.99         | 1.82         | 2.80      | 2.82      | 2.92         | 2.95         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 32     | 2.27      | 2.20      | 2.37         | 2.30         | 2.75      | 2.75      | 2.82         | 2.81         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 64     | 2.09      | 2.05      | 2.13         | 2.10         | 2.95      | 2.91      | 2.98         | 2.95         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 128    | 2.02      | 2.01      | 2.04         | 2.04         | 2.99      | 2.98      | 3.01         | 2.99         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 256    | 2.01      | 2.00      | 2.02         | 2.01         | 3.01      | 2.99      | 3.01         | 3.00         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 512    | 2.00      | 2.00      | 2.01         | 2.01         | 3.00      | 3.00      | 3.00         | 3.00         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 1024   | 2.00      | 1.99      | 2.00         | 2.00         | 3.00      | 3.00      | 3.00         | 3.00         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 2048   | 2.00      | 2.00      | 2.00         | 2.00         | 3.00      | 3.00      | 3.00         | 3.00         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 4096   | 2.00      | 2.00      | 2.00         | 2.00         | 3.00      | 3.00      | 3.00         | 3.00         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 8192   | 2.00      | 2.00      | 2.00         | 2.00         | 3.01      | 3.00      | 3.01         | 3.00         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 16384  | 2.00      | 1.99      | 2.00         | 1.99         | 3.00      | 3.00      | 3.00         | 3.00         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 32768  | 2.00      | 2.01      | 2.00         | 2.01         | 3.00      | 3.00      | 3.00         | 3.00         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 65536  | 2.00      | 2.00      | 2.00         | 2.00         | 3.00      | 3.00      | 3.00         | 3.00         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 131072 | 2.00      | 2.00      | 2.00         | 2.00         | 3.00      | 3.00      | 3.00         | 3.00         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+

*******************************************
3 Project a non-smooth function to FE space
*******************************************
Preform the same projection for the following non-smooth function:

.. math::
   f(x) = \begin{cases} 1 & 0.35 \leq x \leq 0.65 \\ 0 & \text{otherwise} \end{cases}

For which the the main code can be found in `3 ProjectionFE.py <https://github.com/AJJLagerweij/FEM/blob/main/Homework-2/3_Projection_FE.py>`_

.. figure:: ../../Homework-2/images/NonSmooth_Linear_Elements.svg
    :name: NonSmooth_Linear_Elements
    :align: center
    :width: 600

    : Approximating the discrete function with a finite element projections with :math:`N` linear elements is not
    improving with a refined mesh. The spikes around the step change keep the same height, although the width is
    reducing.

.. figure:: ../../Homework-2/images/NonSmooth_Quadratic_Elements.svg
    :name: NonSmooth_Quadratic_Elements
    :align: center
    :width: 600

    : Moving to quadratic elements make it even worse, the spikes at the step change get higher.

.. figure:: ../../Homework-2/images/NonSmooth_E1_vs_Elements.svg
    :name: NonSmooth_E1_vs_Elements
    :align: center
    :width: 600

    : Comparing error 1 to the number of elements shows faster convergence for the linear elements.
    The order seems to be 1 and less then 1 respectively.

.. figure:: ../../Homework-2/images/NonSmooth_E2_vs_Elements.svg
    :name: NonSmooth_E2_vs_Elements
    :align: center
    :width: 600

    : Both approximations seem to be equally bad.
    The order seems to be less then 1.

.. figure:: ../../Homework-2/images/NonSmooth_E1_vs_DOFs.svg
    :name: NonSmooth_E1_vs_DOFs
    :align: center
    :width: 600

    : The linear approximation is better then the quadratic one.
    The order seems to be 1 and less then 1 respectively.

.. figure:: ../../Homework-2/images/NonSmooth_E2_vs_DOFs.svg
    :name: NonSmooth_E2_vs_DOFs
    :align: center
    :width: 600

    : Both approximations seem to be equally bad.
    The order seems to be less then 1.

+--------+-----------------------------------------------------+-----------------------------------------------------+
|        |                        Linear                       |                      Quadratic                      |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| N      | E1 p(1/N) | E2 p(1/N) | E1 p(1/DoFs) | E2 p(1/DoFs) | E1 p(1/N) | E2 p(1/N) | E1 p(1/DoFs) | E2 p(1/DoFs) |
+========+===========+===========+==============+==============+===========+===========+==============+==============+
| 4      | 0.95      | 0.53      | 1.29         | 0.72         | 0.72      | 0.37      | 0.85         | 0.44         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 8      | 0.27      | 1.39      | 0.31         | 1.64         | 0.21      | 1.02      | 0.22         | 1.11         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 16     | 1.41      | 0.34      | 1.53         | 0.37         | 0.70      | -0.01     | 0.73         | -0.01        |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 32     | 0.47      | 1.64      | 0.49         | 1.71         | 0.30      | 1.01      | 0.30         | 1.04         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 64     | 1.51      | 0.35      | 1.55         | 0.36         | 0.70      | -0.01     | 0.71         | -0.01        |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 128    | 0.49      | 1.65      | 0.49         | 1.66         | 0.30      | 1.01      | 0.30         | 1.02         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 256    | 1.52      | 0.35      | 1.53         | 0.35         | 0.70      | -0.01     | 0.71         | -0.01        |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 512    | 0.48      | 1.65      | 0.48         | 1.65         | 0.29      | 1.01      | 0.29         | 1.01         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 1024   | 1.52      | 0.35      | 1.52         | 0.35         | 0.71      | -0.01     | 0.71         | -0.01        |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 2048   | 0.48      | 1.64      | 0.48         | 1.65         | 0.29      | 1.02      | 0.29         | 1.02         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 4096   | 1.52      | 0.36      | 1.52         | 0.36         | 0.71      | -0.01     | 0.71         | -0.01        |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 8192   | 0.48      | 1.64      | 0.48         | 1.64         | 0.29      | 1.01      | 0.29         | 1.01         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 16384  | 1.51      | 0.35      | 1.51         | 0.35         | 0.70      | -0.02     | 0.70         | -0.02        |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 32768  | 0.49      | 1.64      | 0.49         | 1.64         | 0.31      | 1.01      | 0.31         | 1.01         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 65536  | 1.53      | 0.38      | 1.53         | 0.38         | 0.72      | 0.02      | 0.72         | 0.02         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+
| 131072 | 0.46      | 1.64      | 0.46         | 1.64         | 0.25      | 1.06      | 0.25         | 1.06         |
+--------+-----------+-----------+--------------+--------------+-----------+-----------+--------------+--------------+

***************************************
4 Solve Advection-Diffusion PDE with FE
***************************************
Consider the one-dimensional advection diffusion equation:

.. math::
   u_t + u_x - \mu u_{xx} =0 \qquad \forall \qquad x\in\Omega=[0,2\pi]

where :math:`\mu>0` is a coefficient.
Consider periodic boundary conditions and the following initial conditions:

.. math::
   u(x, 0) = \sin^4 (x)

The exact solution to this equation is given by:

.. math::
   u(x,t) = \frac{3}{8} - \frac{1}{2} \exp^{-4\mu t} \cos(2(x-t)) + \frac{1}{8} \exp^{-16\mu t}\cos(4(x-t))

4.1 Solve through FEM
=====================
Solve this problem using a FEM implementation with the following steps:

1. Consider continuous piecewise linear polynomials and interpolatory basis functions.
    Done, see :func:`element.shape1d`.
2. Obtain the discrete weak formulation.
    We need two steps here, firstly we need to project the initial condition, for which the weak form is derived in :func:`pde.projection`. Secondly the PDE will be solved using the method of lines, see :func:`solvers.forwardEuler` and :func:`solvers.backwardEuler`, which needs to be fed with the weak form of the PDE, avalible at :func:`pde.advectivediffusive`.
3. Identify the different matrices associated with the finite element discretization.
    For these functions the Mass (fem.element_mass), Transport (fem.element_transport) and Stiffness (fem.element_stiffness) matrices need to be obtained.
4. Implement and solve the equation via finite elements up to :math:`t = 2\pi`.
    Done, the code in Done, the main code, in `4_AdvectionDiffusion.py <https://github.com/AJJLagerweij/FEM/blob/main/Homework-2/4_AdvectionDiffusion.py>`_, produces :numref:`AdvectionDiffusion_16`, :numref:`AdvectionDiffusion_32`, :numref:`AdvectionDiffusion_64` and :numref:`AdvectionDiffusion_128`.

.. figure:: ../../Homework-2/images/AdvectionDiffusion_16.svg
    :name: AdvectionDiffusion_16
    :align: center
    :width: 600

    : In this course grid large differences between the FD and FE methods can be observed. The forward and backward
    scheme preform nearly the same.

.. figure:: ../../Homework-2/images/AdvectionDiffusion_32.svg
    :name: AdvectionDiffusion_32
    :align: center
    :width: 600

    : With a finer grid and time step the differences become smaller.

.. figure:: ../../Homework-2/images/AdvectionDiffusion_64.svg
    :name: AdvectionDiffusion_64
    :align: center
    :width: 600

    : And smaller.

.. figure:: ../../Homework-2/images/AdvectionDiffusion_128.svg
    :name: AdvectionDiffusion_128
    :align: center
    :width: 600

    : At the finished mesh and time step the results become quite close to the exact solution.

4.2 Compute the error
=====================
Compute the errors :math:`E_1` and :math:`E_2` and compare the results to those of previous weeks homework, in which the same PDE was solved using a Finite Difference approach.
Preform a convergence test as described in :ref:`2.2 Projection`.

.. figure:: ../../Homework-2/images/AdvectionDiffusion_E1.svg
    :name: AdvectionDiffusion_E1
    :align: center
    :width: 600

    : The finite element method seems to converge faster with respect to error 1. This must come from the change in mass matrix, as the stiffness and transport matrix don't differ from the FD method. There does not seem to be a difference between the forward
    and backwards methods, because the time steps are small enough for the forward method to be stable.

.. figure:: ../../Homework-2/images/AdvectionDiffusion_E2.svg
    :name: AdvectionDiffusion_E2
    :align: center
    :width: 600

    : The behaviour of :math:`E_2` is similar to that of :math:`E1`.

+-----+----------+---------------+---------------+---------------+---------------+
|     |          |  FD forward   |  FD backward  |  FE forward   |  FE backward  |
+-----+----------+-------+-------+-------+-------+-------+-------+-------+-------+
| N   | dt       | p E1  | p E2  | p E1  | p E2  | p E1  | p E2  | p E1  | p E2  |
+=====+==========+=======+=======+=======+=======+=======+=======+=======+=======+
| 4   | 6.25E-04 | -1.65 | -1.87 | -1.65 | -1.87 |  4.41 |  4.36 |  4.41 |  4.36 |
+-----+----------+-------+-------+-------+-------+-------+-------+-------+-------+
| 8   | 1.56E-04 | -2.06 | -1.86 | -2.06 | -1.85 | -1.55 | -1.66 | -1.55 | -1.66 |
+-----+----------+-------+-------+-------+-------+-------+-------+-------+-------+
| 16  | 3.91E-05 |  0.56 |  0.35 |  0.56 |  0.35 |  2.08 |  2.19 |  2.08 |  2.19 |
+-----+----------+-------+-------+-------+-------+-------+-------+-------+-------+
| 32  | 9.77E-06 |  1.66 |  1.61 |  1.66 |  1.61 |  3.52 |  3.40 |  3.50 |  3.39 |
+-----+----------+-------+-------+-------+-------+-------+-------+-------+-------+
| 64  | 2.44E-06 |  1.90 |  1.84 |  1.90 |  1.84 |  2.33 |  2.29 |  2.33 |  2.28 |
+-----+----------+-------+-------+-------+-------+-------+-------+-------+-------+
| 128 | 6.10E-07 |  1.99 |  1.99 |  1.99 |  1.99 |  2.02 |  2.02 |  2.02 |  2.02 |
+-----+----------+-------+-------+-------+-------+-------+-------+-------+-------+
