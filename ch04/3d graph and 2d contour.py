# -*- coding: utf-8 -*-
"""
Created on Sun May  3 21:08:19 2020

@author: Han
"""

import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


x = np.arange(-2, 2.5, 0.25)
y = np.arange(-2, 2.5, 0.25)
X, Y = np.meshgrid(x, y)
Z=X**2+Y**2

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z)
plt.show()
plt.contour(X,Y,Z)
plt.show()

Z=(1/5)*X**2+Y**2
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z)
plt.show()
plt.contour(X,Y,Z)
plt.show()

Z=np.sqrt(9-X**2-Y**2)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z)
plt.show()
plt.contour(X,Y,Z)
plt.show()

Z=np.sin(X*Y)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z)
plt.show()
plt.contour(X,Y,Z)
plt.show()