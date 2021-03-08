"""
2D shape functions.

Bram Lagerweij
COHMAS Mechanical Engineering KAUST
2021
"""

# Load external modules.
import matplotlib.pyplot as plt
import numpy as np

# Load my own modules.
from element import shape1d


def quad(x, order):
    phi_x, dphi_x = shape1d(x[0], order)
    phi_y, dphi_y = shape1d(x[1], order)
    dofe = len(phi_x)

    phi2d = np.zeros((dofe*dofe, len(x[1])))
    ij = 0
    for i in range(dofe):
        phi_xi = phi_x[i]
        for j in range(dofe):
            phi_yj = phi_y[j]

            # Obtain product.
            phi2d[ij, :] = phi_xi * phi_yj
            ij += 1

    return phi2d


# Create mesh
num = 10
xi = np.linspace(0, 1, num)
yj = np.linspace(0, 1, num)
xx, yy = np.meshgrid(xi, yj)
x = np.vstack((xx.ravel(), yy.ravel()))

# Obtain values of the shape functions at locations x.
order = 1
phi = quad(x, order)

# Plotting the result.
fig = plt.figure(num='Linear Quads')
for i in range(len(phi)):
    ax = fig.add_subplot(order+1, order+1, i+1, projection='3d')
    ax.set_xlim3d(left=0, right=1)
    ax.set_ylim3d(bottom=0, top=1)
    ax.set_zlim3d(bottom=0, top=1)

    surf = ax.plot_surface(xx, yy, phi[i].reshape(num, num),
                           linewidth=0, antialiased=False)


# Obtain values of the shape functions at locations x.
order = 2
phi = quad(x, order)

# Plotting the result.
fig = plt.figure(num='Quadratic Quads')
for i in range(len(phi)):
    ax = fig.add_subplot(order+1, order+1, i+1, projection='3d')
    ax.set_xlim3d(left=0, right=1)
    ax.set_ylim3d(bottom=0, top=1)
    ax.set_zlim3d(bottom=0, top=1)

    surf = ax.plot_surface(xx, yy, phi[i].reshape(num, num),
                           linewidth=0, antialiased=False)
