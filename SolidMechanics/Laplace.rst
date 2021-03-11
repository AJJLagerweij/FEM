.. |br| raw:: html

   <br />

.. math::
    \require{physics}
    \renewcommand{\vec}[1]{\underline{#1}}

################
Poisson Equation
################

.. admonition:: Toppic

   The Poisson equation is the simplest example of the PDE's considerd in Solid Mechanics.
   It is an eliptical PDE, and is simplified compared to linear elasticity in the sense that its solution is a scalar field, instead fo the vector field found in elasticity problems.
   This makes Poisson's equation a good start to explore numerical solving strategies for Solid Mechanics problems.

   Bram Lagerweij |br|
   11 Feb 2020

.. contents:: Table of Contents
    :local:
    :depth: 2

******************
1 Laplace Equation
******************
The most basic description of the Laplace equation is given by:

.. math::
    \grad^2 u(\vec{m}) &= \pdv[2]{u}{x} + \pdv[2]{u}{y} = 0 \qquad& \forall \vec{m} \in \Omega \\
    \quad\text{s.t.:}& \quad u(\vec{m}) = \vec{\tilde{u}}(\vec{m}) & \forall \vec{m} \in \mathcal{S}_u\\
                     & \quad \grad {u}(\vec{m}) = \tilde{\vec{t}}(\vec{m}) & \forall \vec{m} \in \mathcal{S}_t

Where the entirety of the boundary :math:`\partial\Omega` is the union of these to boundary conditions that do not intersect.

.. math:: 
    \partial\Omega = \mathcal{S}_u \cup \mathcal{S}_t \\
    0 = \mathcal{S}_u \cap \mathcal{S}_t

The following images summarizes this.

.. figure:: ../../Solid Mechanics/images/Domain.svg
   :name: Laplace_Domain
   :align: center
   :width: 250

   A domain :math:`\Omega` subjected to the Laplace equation with combined boundary conditions.

******************
2 Poisson equation
******************
In case of non-homogeneous formulations the Laplace equations is called the Poisson equation.

.. math::
    \grad^2 u(\vec{m}) &= \pdv[2]{u}{x} + \pdv[2]{u}{y} = \vec{b}(\vec{m}) \qquad& \forall \vec{m} \in \Omega \\
    \quad\text{s.t.:}& \quad u(\vec{m}) = \vec{\tilde{u}}(\vec{m}) & \forall \vec{m} \in \mathcal{S}_u\\
                     & \quad \grad {u}(\vec{m}) = \tilde{\vec{t}}(\vec{m}) & \forall \vec{m} \in \mathcal{S}_t

The boundary condition can still be defined in the same way as in the Laplace equation.
An example of such a Poisson problem in 1D is a statically determinate Euler-Bernoulli beam problem.
Solving a these linear beam problem can be done with finite differences.

The PDE described by

.. math::
    EI u''(x) = M(x) &  \qquad \forall x \in\Omega = [0, L]\\

Where :math:`M` is the internal bending moment of the beam. This beam has a length :math:`L` and a stiffness EI.
In general these kinds of problems can not be solved directly in this way, as it is not always possible to describe
the moment explicitly, but because our cantilever beam is statically determinate it can be done.
Now we'll be exploring two examples to introduce the different types of boundary conditions.

Example 1: Dirichlet
####################
.. figure:: ../../Solid Mechanics/images/Simple_Beam_Drawing.svg
    :name: Simply supported beam
    :align: center
    :width: 500

    A beam that is simply supported at :math:`x=0` and :math:`250` mm and subjected to a point load.

In this example we consider a beam with a length of 1000mm which is simply supported at :math:`x=0` and :math:`x=250`.
Simply supported means that the displacement :math:`u` at those points is fixed and equals 0. That is our ODE becomes:

.. math::
    EI \, u''(x) = & M(x)   \quad \forall \quad 0 \leq x \leq 1000\\
    \text{where:} \quad & M(x) =  \begin{cases} -3Px & 0 \leq x \leq L/4\\ P(x - L) & L/4 \leq x \leq L \end{cases}\\
    \text{s.t.:} \quad &u(0) = 0 \\
    & u(L/4) = 0

where I did compute the moment equation explicitly already.
To derive :math:`u''` a central difference scheme is used,

.. math::
    u''(x) = \frac{u(x-dx) - 2 u(x) + u(x+dx)}{dx^2}

We'll be evaluating this derivative an :math:`N` regularly distributed points in our domain.
And if we note :math:`x_n` as the location of one of these points than we can note the derivative as:

.. math::
    u''(x_n) = \frac{u(x_{n-1}) - 2 u(x_n) + u(x_{n+1})}{dx^2}

This is implemented into a matrix format by :func:`finitedifference.Dxx`, such that:

.. math::
    u'' = D_{xx} u

where :math:`u` is a vector with the field at all the discrete points and :math:`u''` the derivative that was calculated.
This does however not yet specify the way to analyze the derivative at the first and last points. After all that would
require the calculation of :math:`u` outside the domain. As a result the matrix will have an empty first and last row.

This and the right hand side (:math:`f`) of the Poisson equation are available through :func:`finitedifference.poisson`.
You would expect that we can solve the system of equations:

.. math::
    EI\,D_{xx}\, u = f

but that is not true, as we'll have to deal with the boundary conditions as well, without those the problem is singular.
To be specific we know that :math:`u(0)=0` and :math:`u(L/4)=0`, this can be used to make the problem determinate.
Lets say that :math:`x_0 = 0` and :math:`x_n = L/4` then we can add the following to equations to our system of equations:

.. math::
    u_0 = 0 \qq{and} u_n = 0

these two equations can be placed in the still empty first and last row of our stiffness matrix and right hand side.
That is in the first row we make the first element equal to 1 and the rest all equal to 0. Similarly the right hand side
of the first degree of freedom is set to 0.
In the last row we set the degree of freedom that corresponds to :math:`x_n` to 1 and the rest to 0, here we do also
set the right hand side of the last row equal to zero (see lines 53 to 61 in the code below).

.. figure:: ../../Solid Mechanics/images/Simply_Solution.svg
    :name: Simply supported solution
    :align: center
    :width: 800

    The finite difference solution of the beam problem seems to be in good agreement with the exact result.
    This simulation was run with 101 degrees of freedom.

.. literalinclude:: /../../SolidMechanics/Double-Simple.py
    :lines: 23-24, 27-81, 91-108
    :linenos:

Example 2: Dirichlet and Neumann
################################
.. figure:: ../../Solid Mechanics/images/Cantilever_Drawing.svg
    :name: Cantilever beam
    :align: center
    :width: 500

    A cantilever beam is fixed in the wall of the left and subjected to a point load at the right.
    This type of constraint, called an endcast, limits both the displacement and rotation, that is :math:`u(0)=0` and
    :math:`u'(0)=0`.

The approach follows exactly what was described in example 1, except of course the constraints.
Our problem is formulate following:

.. math::
    EI \, u''(x) = M(x) &  \quad \forall \quad 0 \leq x \leq 1000\\
    \text{where:} \quad & M(x) = P(x-L)\\
    \text{s.t.:} \quad &u(0) = 0 \\
    & u'(0) = 0

where the moment did change as well because the loading conditions changed. That is after discritization our system
of equations is represented by:

.. math::
    EI\,D_{xx}\, u = f

Now as for the boundary conditions, for the first row we again fill the first element with a 1 and leave the rest 0.
In the right hand side we set the value of the forcing term equal to zero. As a result the first row reads:

.. math::
    u(x_0) = 0

Now for the Neumann boundary it is a bit more tricky. The derivative :math:`u'(x_0)` can be approximated with a backwards
finite difference:

.. math::
    u'(x_0) = \frac{-u(x_0) + u(x_1}{dx} =0

I'll put this in the last row as that one is not yet populated. That means that we have to
populate the first element of the last row with a -1, the second element of that row with a 1 and set the last element
of the right hand side to zero as well. (see lines 64 to 72 below)

.. figure:: ../../Solid Mechanics/images/Cantilever_Solution.svg
    :name: Cantilever solution
    :align: center
    :width: 800

    The finite difference solution of the beam problem seems to be in good agreement with the exact result.
    This simulation was run with 101 degrees of freedom.

.. literalinclude:: /../../SolidMechanics/Cantilever.py
    :lines: 23-24, 27-65, 69-87
    :linenos:
