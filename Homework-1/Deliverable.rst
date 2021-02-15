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

.. admonition:: Toppic

   Homework regarding the first week. The goal is  to  work with basic numerical approximation of PDE's' and functions.

   Bram Lagerweij |br|
   08 Feb 2020

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

.. figure:: ../../Homework-1/images/AdDiff1.svg
   :name: AdDiff1
   :align: center
   :width: 600

   : The forward difference scheme is unstable for :math:`dt=10^{-3}`, the backward scheme behaves as expected.
   An animated plot can be found :download:`here <../../Homework-1/images/AdDiff1.webm>`.

.. figure:: ../../Homework-1/images/AdDiff2.svg
   :name: AdDiff2
   :align: center
   :width: 600

   : With a timestep of  :math:`dt=10^{-3}` both the forward and backward difference scheme are stable.
   An animated plot can be found :download:`here <../../Homework-1/images/AdDiff2.webm>`.

.. figure:: ../../Homework-1/images/AdDiff3.svg
   :name: AdDiff3
   :align: center
   :width: 600

   : As expected with a timestep of :math:`dt=10^{-4}` both time integrations behave stable.
   An animated plot can be found :download:`here <../../Homework-1/images/AdDiff3.webm>`.


1.2 Advective PDE
=================
Consider :math:`\mu=0` and :math:`c=2` and solve the PDE using the explicit and the implicit methods.
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
    f_h(x) = a_0 + a_1 x + a_2 x^2 + \dots + a_{N-1} x^{N-1}

Consider different levels of refinement, :math:`N=4,\, 5,\, 6,\,\dots,\,10` and for each approximation report both :math:`E_1` and :math:`E_2`.

2.1.1 Taylor series
*******************
The Taylor series till the order :math:`N` is defined through:

.. math::
    f_h(x) = \sum_{n=0}^N \frac{f^{(n)}(x_0)}{n!} (x - x_0)^n

Which imediately got me into problems, analyzing the :math:`n`-th derivative of a function is a numerically a pain.
Quickly the round off errors become significant, and from the 5th derivative onward the basic `scipy` Taylor series function became useless.
As a result I decided to hardcode the weighting contsants in our expansion, these are obtained from manual derivatives.

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

where :math:`P` is the period of the function :math:`f(x)` and :math:`c_n` are complex valued coeffinients that can be found through a Fourier Transform.
In our case I used a FFT algorithm to find these coefficients from our discrete dataset, esentially the rfft tries to solve:

.. math::
    c_n = \sum_{n=0}^{K} x_k \exp^{\frac{2\pi k n}{K-1}} \qquad n = 0, \dots, N

in a highly efficient manner. Notice that for each unknown :math:`c_n` consists of a real and imaginary part.
This does mean that this approximation for any given :math:`N` is more comlex.
The resulting approximation is shown in :numref:`Fourier_series`. Which show that this series is highly efficient in the approximation of our function.
This is not to surpricing, afterall we are approximation a trigonometric functions with a serie of trigonometric functions it is likely that we find the exact function somewhere in our series.

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

This minimization can efficiently be casted to a system of equations and subsequentely be solved.
This system of equations has :math:`N` unknowns and :math:`N` functions, and because each of these functions is linearly independent a solution exists.
Simply said we construct a polynomial that goes exactly through these :math:`N` points.

.. figure:: ../../Homework-1/images/PolyN.svg
   :name: Polynomial_series
   :align: center
   :width: 600

   : Approximating :math:`f(x)` with a polynomials of order :math:`N-1` using :math:`N` sample points.

One can also choose to use more sample points to evaluate the minimization problem, lets consider that we use :math:`M` sample points.
It is not generaly possible to find a :math:`N-1` order polynomial to fit exactly through more then :math:`N` points.
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
For the comparison of these different approximations I've plottet the errors on a log scale.
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

I assume that the error of the Taylor series is increasing because the higher order terms will cause heigher errors at the boundarie of our domain.
But all in all it is my opinion that the Taylor series is a bad approximator for this purpose, it is difficult to calculate due to the derivatives and 
the result is inaccurate. This is not so surpricing however, Taylor series are ment to approximate the behaviour of a function around a given point :math:`x_0`
to characterize the local behaviour. We are here using it on a relatively large domain.

The script used for these computations can be found at :download:`3 LocalApproximation.py <../../Homework-1/3_LocalApproximation.py>`

2.2 Local Approximations
========================
Split the domain :math:`\Omega` into :math:`N` cells. For each cell :math:`K`, compute linear and quadratic approximations :math:`f_K(x)` where :math:`f_K(x_i)=f(x_i)` where :math:`x_i` are evenly spaced gridpoints, including the boundaries of the cell.
Comput and report both :math:`E_1` and :math:`E_2` for a different numbers of cells :math:`N=4,\, 5,\, 6,\,\dots,\,10`.
