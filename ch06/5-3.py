# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from common.optimizer import *

def f(x, y):
    return x**2 / 20.0 + y**2

fig = plt.figure(figsize=(10,5))
axs = fig.subplots(1,2,subplot_kw={'projection':'3d', 'title':'coarser mesh'})
X = np.arange(-10,10,1)
Y = np.arange(-5,5,1)
X, Y = np.meshgrid(X,Y)
Z = f(X,Y)
_=axs[0].plot_surface(X,Y,Z)
X = np.arange(-10,10,0.01)
Y = np.arange(-5,5,0.01)
X, Y = np.meshgrid(X,Y)
Z = f(X,Y)
_=axs[1].plot_surface(X,Y,Z)
plt.show()