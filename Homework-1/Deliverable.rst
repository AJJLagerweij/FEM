.. |br| raw:: html

   <br />

.. math::
    \require{physics}
    \renewcommand{\vec}[1]{\underline{#1}}
    \renewcommand{\exp}{\text{e}}
    \DeclareMathOperator*{\argmin}{argmin}

##########
Homework 1
##########

.. admonition:: Topic

   Homework regarding the first week. The goal is  to  work with basic numerical approximation of PDE's' and functions.

   Bram Lagerweij |br|
   18 Feb 2021


.. contents:: Table of Contents
    :local:
    :depth: 2

*****************
1 Method of Lines
*****************
Consider the one-dimensional advection diffusion equation:

.. math::
	{u}_{t} + c{u}_{x} - \mu{u}_{xx} = 0 \qquad \forall \, x \in \Omega = [0, 1] \quad \& \quad t>0

where :math:`\mu>0` is the diffusion coefficient and :math:`c` the wave speed. Consider periodic boundary conditions and the following initial condition:

.. math::
	u(x,0) = \sin(2\pi x)

What do we expect the exact solution to do? Due to the advective part, the initial condition travels at constant speed to the right.
At the same time, due to the diffusive term, the initial condition is dissipated at a rate that depends on :math:`\mu`.

Consider the following discretization. Use second-order central finite differences to approximate :math:`u_x` and :math:`u_{xx}`.
Use forward and backward Euler to obtain full discretization (write down the schemes). Consider a fixed mesh with of :math:`\Delta x`.

1.1 Advective Diffusive PDE
===========================
Consider a final time of :math:`t=1`, :math:`c=1` and :math:`\mu=0.01`. For each full discretization proceed as follows:

1. Experiment using the following time step sizes: :math:`\Delta t = 10^{−4},\, 10^{−3}` and :math:`10^{−2}`. 
2. How do the explicit and implicit methods behave for these time steps?

There is a so called Courant-Friedrichs-Lewy condition that formulates a condition of stability on the model:

.. math::
    C = \frac{c\Delta t}{\Delta x} \leq C_{\max}

Where :math:`C_{\max}` is a constant, which for explicit schemes, such as forward Euler, is around 1.
If the condition is violated the method becomes unstable, that does not mean that the results are unstable from the first iteration.

.. figure:: ../../Homework-1/images/AdDiff1.svg
   :name: AdDiff1
   :align: center
   :width: 600

   : The forward difference scheme is unstable for :math:`dt=10^{-2}`, the backward scheme behaves as expected.
   `Click here for an animated version <_static/AdDiff1.webm>`__.

.. figure:: ../../Homework-1/images/AdDiff2.svg
   :name: AdDiff2
   :align: center
   :width: 600

   : With a timestep of  :math:`dt=10^{-3}` both the forward and backward Euler scheme are stable.
   `Click here for an animated version <_static/AdDiff2.webm>`__.

.. figure:: ../../Homework-1/images/AdDiff3.svg
   :name: AdDiff3
   :align: center
   :width: 600

   : As expected with a timestep of :math:`dt=10^{-4}` both time integrations behave stable.
   `Click here for an animated version <_static/AdDiff3.webm>`__.

.. literalinclude:: /../../Homework-1/1_AdvectiveDiffusive.py
   :linenos:

1.2 Advective PDE
=================
Consider :math:`\mu=0` and :math:`c=2` and solve the PDE using the explicit and the implicit methods.
Use :math:`\Delta t = 10^{−4}` and solve the problem for the following final times :math:`t=1,\, 5,\, 10,\, 15` and :math:`20`.
Comment on the behaviour of each full discretization as the final time increases.

.. figure:: ../../Homework-1/images/AdvectUnstable.svg
   :name: AdvectiveUnstable
   :align: center
   :width: 600

   : Even with small time steps this type of hyperbolic like equation can become unstable when using a forward Euler method.
   `Click here for an animated version <_static/AdvectUnstable.webm>`__.

Due to the region of convergence of the forward Euler method such a hyperbolic PDE with no dissipation will always be unstable.
In the animation the instabilities become only clear after 14 seconds. Nevertheless, even at :math:`t=1` the method should be considered unstable.
Similarly the backward Euler is inaccurate as well, it is too dissipative, after 20 seconds around 20% of our, wave magnitude has disappeared.

.. literalinclude:: /../../Homework-1/2_Advective.py
   :linenos:

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
    f_h(x) = a_0 + a_1 x + a_2 x^2 + \dots + a_{N-1} x^{N-1}

Consider different levels of refinement, :math:`N=4,\, 5,\, 6,\,\dots,\,10` and for each approximation report both :math:`E_1` and :math:`E_2`.

2.1.1 Taylor series
*******************
The Taylor series till the order :math:`N` is defined through:

.. math::
    f_h(x) = \sum_{n=0}^N \frac{f^{(n)}(x_0)}{n!} (x - x_0)^n

Which immediately got me into problems, analyzing the :math:`n`-th derivative of a function is a numerically a pain.
Quickly the round off errors become significant, and from the 5th derivative onward the basic `scipy` Taylor series function became useless.
As a result I decided to hardcode the weighting constants in our expansion, these are obtained from manual derivatives.

.. figure:: ../../Homework-1/images/Taylor.svg
   :name: Taylor_Expansion
   :align: center
   :width: 600

   : Approximating :math:`f(x)` with a Taylor series centered around :math:`x_0=0.5` till order 10.

From :numref:`Taylor_expansion` it can be observed that the Taylor series is not a very efficient approximation.
At the boundary of our domain the error is very high.

2.1.2 Fourier series
********************
The Fourier series, which we assume to be real, approximates the equation with:

.. math::
    f_h(x) = \sum_{n=0}^N  c_n\exp^{\frac{2\pi n x}{P}i} + \bar{c}_n\exp^{-\frac{2\pi n x}{P}i}

where :math:`P` is the period of the function :math:`f(x)` and :math:`c_n` are complex valued coefficients that can be found through a Fourier Transform.
In our case I used a FFT algorithm to find these coefficients from our discrete dataset, essentially the real-FFH tries to solve:

.. math::
    c_n = \sum_{n=0}^{K} x_k \exp^{\frac{2\pi k n}{K-1}} \qquad n = 0, \dots, N

in a highly efficient manner. Notice that for each unknown :math:`c_n` consists of a real and imaginary part.
This does mean that this approximation for any given :math:`N` is more complex.
The resulting approximation is shown in :numref:`Fourier_series`. Which show that this series is highly efficient in the approximation of our function.
This is not to surprising, after all we are approximation a trigonometric functions with a series of trigonometric functions it is likely that we find the exact function somewhere in our series.

.. figure:: ../../Homework-1/images/Fourier.svg
   :name: Fourier_series
   :align: center
   :width: 600

   : Approximating :math:`f(x)` with a Fourier series seems to be exact from the fourth order.

2.1.3 Polynomial series
***********************
The polynomial series 

.. math::
    f_h(x) = a_0 + a_1 x + a_2 x^2 + \dots + a_{N-1} x^{N-1}

was to be found with a fitting through :math:`N` evenly spaced points :math:`x_i` throughout this interval.
It should be noted that this type of fitting can be rewritten as an minimization:

.. math::
    \argmin_{a_0, \dots, a_{N-1}} \sum_{i=0}^N \big( f(x_i) - f_h(x_i) \big)^2 
.. math::
    \qq{that means: find} a_0, \dots a_{N-1} \qq{such that} f(x_i) - f_h(x_i) = 0 \quad \forall x_i

This minimization can efficiently be casted to a system of equations and subsequently be solved.
This system of equations has :math:`N` unknowns and :math:`N` functions, and because each of these functions is linearly independent a solution exists.
Simply said we construct a polynomial that goes exactly through these :math:`N` points.

.. figure:: ../../Homework-1/images/PolyN.svg
   :name: Polynomial_series
   :align: center
   :width: 600

   : Approximating :math:`f(x)` with a polynomials of order :math:`N-1` using :math:`N` sample points.

One can also choose to use more sample points to evaluate the minimization problem, lets consider that we use :math:`M` sample points.
It is not generally possible to find a :math:`N-1` order polynomial to fit exactly through more then :math:`N` points.
But we can find the best polynomial, to be specific one that minimizes:

.. math::
    \argmin_{a_0, \dots, a_{N-1}} \sum_{i=0}^M \big( f(x_i) - f_h(x_i) \big)^2

Which is as if we are minimizing our error :math:`E_2` at only discrete points, instead of solving the integral itself.
Anyway, :numref:`Polynomial5_series` shows this fit would look like.
The results seems closer, because we're not just minimizing the error at :math:`N` points but at :math:`5N` points.

.. figure:: ../../Homework-1/images/Poly5N.svg
   :name: Polynomial5_series
   :align: center
   :width: 600

   : Approximating :math:`f(x)` with a polynomials of order :math:`N-1` using :math:`M=5N` sample points.

2.1.4 Comparison
****************
For the comparison of these different approximations I've plotted the errors on a log scale.
Please do note that the Fourier series has 2 times as many unknowns for the :math:`N` compared to the other methods.

.. figure:: ../../Homework-1/images/E1.svg
   :name: Error1
   :align: center
   :width: 600

   : The error :math:`E_1` for our different approximations where the approximation order ranges from 1 to 20.

.. figure:: ../../Homework-1/images/E2.svg
   :name: Error2
   :align: center
   :width: 600

   : The error :math:`E_2` for our different approximations where the approximation order ranges from 1 to 20.

I assume that the error of the Taylor series is increasing because the higher order terms will cause higher errors at the boundaries of our domain.
But all in all it is my opinion that the Taylor series is a bad approximation for this purpose, it is difficult to calculate due to the derivatives and
the result is inaccurate. This is not so surprising however, Taylor series are meant to approximate the behaviour of a function around a given point :math:`x_0`
to characterize the local behaviour. We are here using it on a relatively large domain.

The script used for these computations can be found at `3 GlobalApproximation.py <https://github.com/AJJLagerweij/FEM/blob/main/Homework-1/3_GlobalApproximation.py>`_.

2.2 Local Approximations
========================
Split the domain :math:`\Omega` into :math:`N` cells. For each cell :math:`K`, compute linear and quadratic approximations :math:`f_K(x)` where :math:`f_K(x_i)=f(x_i)` where :math:`x_i` are evenly spaced gridpoints, including the boundaries of the cell.
Compute and report both :math:`E_1` and :math:`E_2` for a different numbers of cells :math:`N=4,\, 5,\, 6,\,\dots,\,10`.

The approximation by linear elements is created by scaling hat (shape) functions appropriately.
These functions are chosen in such a way that:

1. The sum of all the shape functions together equals one, :math:`\sum_{n=1}^{N} \varphi_i(x) = 1` This is called the Partition of Unity Method.
2. There where a single function reaches its maximum all the other functions equal zero.

Then our approximation is defined by:

.. math::
    f_h(x) = \sum_{n=1}^N w_n \varphi_n(x)

where the weights :math:`w_n` are unknown. But because the shape function where chosen smartly these weights are independent.
After all at the point where a single shape function reaches its maximum (1) the other functions are zero.
As a result the weight of this shape function equals the value of the function we are trying to approximate at the center point of the shape:

.. math::
    w_n = f(X_n)

where :math:`X_n` denotes the point where shape function :math:`\varphi_n(x)` reaches its maximum.

2.2.1 Linear Elements
*********************
In the case of linear elements these shape functions are defined as:

.. math::
    \varphi_n(x) =
    \begin{cases}
        0 \quad &\forall \quad 0 &\leq x \leq &X_{n-1} \\
        \frac{x - X_{n-1}}{X_{n} - X_{n-1}} \quad &\forall\quad  X_{n-1} &\leq x \leq &X_{n}\\
        1 - \frac{x - X_{n}}{X_{n+1} - X_{n}} \quad &\forall \quad X_{n} &\leq x \leq &X_{n+1}\\
        0 \quad & \forall \quad X_{n+1} &\leq x \leq &L
    \end{cases}

where :math:`X_n` is the node of this shape function, :math:`X_{n-1}` and :math:`X_{n+1}` the nodes surrounding ours.

A more efficient formulation includes the creation of a unit function that is rescaled depending on the locations of
the nodes. But I haven't yet implemented such an function yet.

.. figure:: ../../Homework-1/images/Linear_elements.svg
    :name: Linear Elements
    :align: center
    :width: 600

    : The function :math:`4\sin(\pi x) + 1` approximated with four elements.
    The first element contain the orange and half of the green shape function.

.. figure:: ../../Homework-1/images/Linear.gif
    :name: Linear Elements Refinement
    :align: center
    :width: 600

    : The function :math:`4\sin(\pi x) + 1` approximated more and more linear elements.

.. figure:: ../../Homework-1/images/Linear_ele.svg
    :name: Linear Elements Approxmation
    :align: center
    :width: 600

    : The approximation of :math:`f(x)` with linear elements.


2.2.2 Quadratic Elements
************************
In the case of quadratic elements there are two different types of shape function.
One of these function extents into two elements, similar to what the linear element does.
The second shape function is only inside a single element, and on an interior node.
This node is placed exactly in the middle between the start and end of the element.
I'll give these nodes the subscripts :math:`n-\frac{1}{2}` and :math:`n+\frac{1}{2}`.
Now the shape functions are defined by:

.. math::
    \varphi_n(x) &=
    \begin{cases}
        0 \quad &\forall \quad 0 &\leq x \leq &X_{n-1} \\
        \frac{2}{(X_n - X_{n-1})^2} (x - X_{n-1})(x - X_{n-\frac{1}{2}}) \quad &\forall\quad  X_{n-1} &\leq x \leq &X_{n}\\
        \frac{2}{(X_{n+1} - X_{n})^2}(x - X_{n+1})(x - X_{n+\frac{1}{2}}) \quad &\forall \quad X_{n} &\leq x \leq &X_{n+1}\\
        0 \quad & \forall \quad X_{n+1} &\leq x \leq &L
    \end{cases}\\
    \varphi_{n-\frac{1}{2}} (x) &=
    \begin{cases}
        0 \quad &\forall \quad 0 &\leq x \leq &X_{n-1} \\
        -\frac{4}{(X_n - X_{n-1})^2} (x - X_{n-1})(x - X_{n}) \,\, \quad &\forall\quad  X_{n-1} &\leq x \leq &X_{n}\\
        0 \quad & \forall \quad X_{n+1} &\leq x \leq &L
    \end{cases}

Again a more efficient formulation includes the creation of a unit function that is rescaled depending on the locations of
the nodes. But I haven't yet implemented such an function yet.

.. figure:: ../../Homework-1/images/Quadratic_elements.svg
    :name: Quadratic Elements
    :align: center
    :width: 600

    : The function :math:`4\sin(\pi x) + 1` approximated with four elements.
    The first element contain the orange and half of the green shape function.

.. figure:: ../../Homework-1/images/Quadratic.gif
    :name: Quadratic Elements Refinement
    :align: center
    :width: 600

    : The function :math:`4\sin(\pi x) + 1` approximated more and more quadratic elements.

.. figure:: ../../Homework-1/images/Quadratic_ele.svg
    :name: Quadratic Elements Approximation
    :align: center
    :width: 600

    : The approximation of :math:`f(x)` with quadratic elements.

It is important to notice from :numref:`Quadratic Elements Approximation` that the resulting curve is not smooth.
for example at :math:`x=0.5` one can see that the red approximation (6 elements) is non-smooth.

2.2.3 Comparison
****************
For the comparison of these different approximations I've plotted the errors on a log scale.
Please do note that the quadratic elements have :math:`(N+1)N` unknowns where the linear elements have :math:`N+1` weights
to be determined.
Nevertheless there is no interdependency between these weights, which as mentioned before means that these can be determined
independently.

.. figure:: ../../Homework-1/images/E1_ele.svg
   :name: Error1_elements
   :align: center
   :width: 600

   : The error :math:`E_1` for our element based approximations with 1 to 20 elements.

.. figure:: ../../Homework-1/images/E2_ele.svg
   :name: Error2_elements
   :align: center
   :width: 600

   : The error :math:`E_2` for our element based approximations with 1 to 20 elements.

The script used for these computations can be found at `4 LocalApproximation.py <https://github.com/AJJLagerweij/FEM/blob/main/Homework-1/4_LocalApproximation.py>`_.
