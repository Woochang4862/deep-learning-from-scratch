import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import Relu

X = np.array([[1,-2,3,-4],
              [-5,6,-7,8]])
dout = np.array([[1,-2,-3,4],
                 [-1,2,3,-4]])

relu = Relu()
Y = relu.forward(X)
dout = relu.backward(dout)
print(dout)