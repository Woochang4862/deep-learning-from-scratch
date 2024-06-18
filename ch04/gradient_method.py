# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])    

plt.figure(figsize=(12,4))
step_num = 20

x, x_history = gradient_descent(function_2, init_x, lr=0.01, step_num=step_num)
print(x_history)
plt.subplot(1,3,1)
plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.title("learning rate : 0.01")
plt.plot(x_history[:,0], x_history[:,1], 'o')

x, x_history = gradient_descent(function_2, init_x, lr=1.01, step_num=step_num)
print(x_history)
plt.subplot(1,3,2)
plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.title("learning rate : 1.01")
plt.plot(x_history[:,0], x_history[:,1], 'o')

x, x_history = gradient_descent(function_2, init_x, lr=0.1, step_num=step_num)
print(x_history)
plt.subplot(1,3,3)
plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.title("learning rate : 0.1")
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.show()
