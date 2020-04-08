import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar as minimize_scalar
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
data = pd.read_csv('weights_heights.csv', index_col='Index')
x = np.array(data['Weight'])
y = np.array(data['Height'])
n = x.__len__()


def get_error(w0, w1):
    s = 0
    for i in range(0, n):
        s = s + (y[i] - (w0 + w1 * x[i])) ** 2
    return s


def f(w0, w1, x):
    return w0 + w1 * x


w0, w1 = 60, 0.05
data.plot.scatter(x='Weight', y='Height')
# s = np.arange(25000)
# for i in range(0, n):
#    s[i] = f(w0, w1, x[i])
#    print(s[i])

#plt.plot(x, f(w0, w1, x), color='orange')
w0, w1 = 50, 0.16
#plt.plot(x, f(w0, w1, x), color='orange')
s=[]
w1=np.linspace(0, 0.36, 100)
s=[get_error(50, w2) for w2 in w1]
#plt.plot(w1, s, color='blue')

min=minimize_scalar(lambda w1:get_error(50, w1), bounds=(-5, 5)).x
print(min)

#plt.plot(x, f(50, min, x), color='green')
fig = plt.figure()
ax = fig.gca(projection='3d')

w0 = np.arange(-100, 101, 1)
w1 = np.arange(-5, 5, 0.25)
w0, w1 = np.meshgrid(w0, w1)
Z = get_error(w0, w1)

surf = ax.plot_surface(w0, w1, Z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#minimize(get_error, (0, 0), method='L-BFGS-B', bounds = ((-100, 100),(-5, 5)))
w00, w11 = minimize(lambda w: get_error(w[0], w[1]), (0, 0), bounds = ((-100, 100), (-5, 5)), method='L-BFGS-B').x
print (w00, w11)

data.plot.scatter(x='Weight', y='Height')
plt.plot(x, f(w00, w11, x), color='red')
plt.show()
