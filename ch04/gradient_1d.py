# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

"""
중앙차분을 이용하여 1변수 함수 수치(aka 근사) 미분
@param:
    f : 미분할 함수
    x : 미분할 지점
"""
def numerical_diff(f, x):
    h = 1e-4 # 0.0001 :: 활용분야에 따라서 얼마나 정교하게할지 
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x # df = 0.02*x + 0.1


def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    def line(t):
        return d*(t-x) + f(x)
    return line
     
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
