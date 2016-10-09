import matplotlib.pyplot as plt
import numpy as np
import math

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#Q1
X1 = np.random.normal(4,2,100)
X2 = np.random.normal(3,3,100) + 0.5*X1
print("The mean of X1 is {}".format(np.mean(X1)))
print("The mean of X2 is {}".format(np.mean(X2)))


#this command shows the covariance of X1 and X2
print("the cov matrix")
print(np.cov(X1, X2))

#this command shows tha eigenvalues and the eigenvectors
print("The eigenvalues and the eigenvectors")
print(np.linalg.eig(np.cov(X1, X2)))
plt.figure()
#this code block plot d i)
plt.plot(X1, X2, "ro")
plt.axis([-15,15,-15,15])

#this code block plot problem d ii)
eigvArray, eigv = np.linalg.eig(np.cov(X1, X2))

eigv1 = eigv[:, 0] * eigvArray[0] #stretch to the correct length
eigv2 = eigv[:, 1] * eigvArray[1]
ax = plt.axes()

ax.arrow(np.mean(X1), np.mean(X2), eigv1[0], eigv1[1], head_width=0.5, head_length=0.4, fc='k', ec='k')
ax.arrow(np.mean(X1), np.mean(X2), eigv2[0], eigv2[1], head_width=0.5, head_length=0.4, fc='k', ec='k')

#this block for problem e)
X1_meam = np.mean(X1)
X1 = [i - X1_meam for i in X1]
X2_mean = np.mean(X2)
X2 = [i - X2_mean for i in X2]
k = [[X1[i], X2[i]] for i in range(100)]
M = [eigv[:, 1], eigv[:, 0]]
k = [np.dot(M, np.transpose(k[i])) for i in range(100)]
rX1 = [i[0] for i in k]
rX2 = [i[1] for i in k]
plt.plot(rX1, rX2, "bo")
plt.axis([-15,15,-15,15])
plt.title("the red dots are the original plot and the blue dots are the plot after rotating")
plt.show()

