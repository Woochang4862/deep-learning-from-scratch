import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import Affine, Relu, SoftmaxWithLoss

x = np.arange(1,6).reshape(1,-1)
t = np.array([[1,0,0,0,0]])

W1 = np.array([[0,0,0,0,1],
               [1,0,0,0,0],
               [0,-1,0,0,0],
               [0,0,-1,0,0],
               [0,0,0,-1,0]])
b1 = np.zeros(5)
W2 = np.array([[5,0,0,0,5],
               [5,0,0,0,5],
               [5,0,0,0,5],
               [5,0,0,0,5],
               [-5,5,5,5,-5]])
b2 = np.zeros(5)

layers = []
layers.append(Affine(W1,b1))
layers.append(Relu())
layers.append(Affine(W2,b2))
softmaxWithLoss = SoftmaxWithLoss()

# forward
for layer in layers:
    x = layer.forward(x)
loss = softmaxWithLoss.forward(x, t)
#print(loss)

# backward
dout = softmaxWithLoss.backward()
for layer in reversed(layers):
    dout = layer.backward(dout)
    if isinstance(layer, Affine):
        print(layer.dW, layer.db)