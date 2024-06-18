import sys, os
sys.path.append(os.pardir)
from common.layers import Affine
import numpy as np

W = np.arange(1,7).reshape(2,-1)
B = np.arange(7,10)
affine = Affine(W,B)
dout = np.array([[2,1,-1],[1,0,0],[0,0,1]])
X = np.array([[1,0],[0,1],[1,1]])
affine.forward(X)
print(affine.backward(dout))
print(affine.dW)
print(affine.db)