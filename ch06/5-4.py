# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from common.optimizer import *

def f(x, y):
    return x**2 / 20.0 + y**2

def df(x, y):
    return {'x':x / 10.0, 'y':2.0*y}

init_pos = (-7.0, 2.0)
params = {'x':init_pos[0],'y':init_pos[1]}

optimizers = dict()
optimizers["SGD"] = SGD(lr=0.95)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
optimizers["Adam"] = Adam(lr=0.3)

fig = plt.figure(figsize=(10,10))
axs = fig.subplots(2,2,subplot_kw={'projection':'3d'}).flatten()
for i, key in enumerate(optimizers.keys()):
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    z_history = []
    params = {'x':init_pos[0],'y':init_pos[1]}

    for _ in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        z_history.append(f(params['x'], params['y']))

        grads = df(params['x'],params['y'])
        optimizer.update(params, grads)

    # X = np.arange(-10,10,0.01)
    # Y = np.arange(-5,5,0.01)
    # X, Y = np.meshgrid(X,Y)
    # Z = f(X,Y) - 10
    _=axs[i].set_title(key)
    # _=axs[i].plot_surface(X,Y,Z)
    _=axs[i].scatter(x_history,y_history,z_history,color='r')
plt.show()