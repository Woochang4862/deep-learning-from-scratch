from functions import my_softmax, softmax
import numpy as np

x = np.array([[1,1,100],[100,1,1],[1,100,1]])
my_sm = my_softmax(x)
sm = softmax(x)
print(f'my_softmax : \n{my_sm}')
print(f'softmax : \n{sm}')
print(f'sum of my_sm : {np.sum(my_sm,axis=1)}')
print(f'sum of sm : {np.sum(sm,axis=1)}')
print(f'{1.01122149e-43+1.01122149e-43+1.00000000e+00}')
