import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'


#(a)
delta = 0.025
x = np.arange(-2.0, 4.0, delta)
y = np.arange(-5.0, 7.0, delta)
X, Y = np.meshgrid(x, y)
Z = mlab.bivariate_normal(X, Y, 1.0, 2.0, 1.0, 1.0)
plt.figure()
CS = plt.contour(X, Y, Z, 8)
plt.axis([-6,6,-6,6])
plt.clabel(CS, fontsize=9, inline=1)
plt.title('(a)')

#(b)
delta = 0.025
x = np.arange(-4.0, 2.0, delta)
y = np.arange(-3.0, 7.0, delta)
X, Y = np.meshgrid(x, y)
Z = mlab.bivariate_normal(X, Y, 2.0, 3.0, -1.0, 2.0, 1.0)
plt.figure()
CS = plt.contour(X, Y, Z, 8)
plt.clabel(CS, fontsize=9, inline=1)
plt.title('(b)')

#(c)
delta = 0.025
x = np.arange(-3.0, 5.0, delta)
y = np.arange(-3.0, 5.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 2.0, 1.0, 0.0, 2.0, 1.0)
Z2 = mlab.bivariate_normal(X, Y, 2.0, 1.0, 2.0, 0.0, 1.0)
# difference of Gaussians
Z = Z1 - Z2
plt.figure()
CS = plt.contour(X, Y, Z, 12)
plt.clabel(CS, fontsize=9, inline=1)
plt.title('(c)')

#(d)
delta = 0.025
x = np.arange(-3.0, 5.0, delta)
y = np.arange(-3.0, 5.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 2.0, 1.0, 0.0, 2.0, 1.0)
Z2 = mlab.bivariate_normal(X, Y, 2.0, 3.0, 2.0, 0.0, 1.0)
# difference of Gaussians
Z = Z1 - Z2
plt.figure()
CS = plt.contour(X, Y, Z, 15)
plt.clabel(CS, fontsize=9, inline=1)
plt.title('(d)')

#(e)
delta = 0.025
x = np.arange(-5.0, 5.0, delta)
y = np.arange(-5.0, 5.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 2.0, 1.0, 1.0, 1.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 2.0, 2.0, -1.0, -1.0, 1.0)
# difference of Gaussians
Z = Z1 - Z2
plt.figure()
CS = plt.contour(X, Y, Z, 12)
plt.clabel(CS, fontsize=9, inline=1)
plt.title('(e)')

plt.show()