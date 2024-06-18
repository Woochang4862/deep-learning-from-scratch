import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import Affine, Sigmoid, SoftmaxWithLoss

x = np.array([[2,1]])
t = np.array([[1,0]])

W1 = np.array([[2,2,1],[2,1,2]])
W1 = np.log(W1)
b1 = np.zeros(3)
W2 = np.array([[9,0],
               [0,5],
               [0,6]])
b2 = np.zeros(2)

layers = []
layers.append(Affine(W1,b1))
layers.append(Sigmoid())
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