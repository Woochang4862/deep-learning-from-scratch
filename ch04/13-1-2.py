import sys, os
sys.path.append(os.pardir)
from common.layers import Sigmoid
import numpy as np

X = np.array([np.log(i) for i in range(2,10)]).reshape(2,-1)
sigmoid = Sigmoid()
Y = sigmoid.forward(X)
dout = np.array([i**2 for i in range(3,11)]).reshape(2,-1)
dout = sigmoid.backward(dout)
print(dout)