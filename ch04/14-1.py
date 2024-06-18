import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
x = np.log([[2,3,5],[7,2,1]])
t = np.array([[0,0,1],[1,0,0]])

y = softmax(x)
print(y)
loss = cross_entropy_error(y, t)
print(loss, -0.5*np.log(0.35))

batch_size = t.shape[0]
print(t.size == y.size)
dx = (y - t) / batch_size
print(dx)

t = np.argmax(t, axis=1)
print(t)

print(t.size == y.size)
dx = y.copy()
dx[np.arange(batch_size), t] -= 1
dx = dx / batch_size
print(dx)