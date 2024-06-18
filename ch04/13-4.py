import sys, os
sys.path.append(os.pardir)
import numpy as np
# from collections import OrderedDict
# from common.layers import Affine, SimpleAffine, Sigmoid, Relu, SoftmaxWithLoss

# class NLayersNet:
#     # N : M+1 Layers Neural Net (M : legnth of hidden_sizes)
#     # input ~ H1 ~ H2 ~ ... ~ HM ~ output
#     def __init__(self, input_size, output_size, hidden_sizes, weight_init_std):
#         self.params = OrderedDict()
#         self.params['W1'] = weight_init_std*np.random.randn(input_size, hidden_sizes[0])
#         self.params['b1'] = np.zeros(hidden_sizes[0])
        
#         for i,hidden_size in enumerate(hidden_sizes[1:],2):
#             self.params[f'W{i}'] = weight_init_std*np.random.randn(hidden_sizes[i-1],hidden_size)
#             self.params[f'b{i}'] = np.zeros(hidden_size)
        
#         self.params[f'W{len(hidden_sizes)+1}'] = weight_init_std*np.random.randn(hidden_sizes[-1],output_size)
#         self.params[f'b{len(hidden_sizes)+1}'] = np.zeros(output_size)

#         self.layers = OrderedDict()
#         params = zip(list(self.params.values())[:-1],list(self.params.values())[1:])
#         for i,(W,b) in enumerate(params,1):
#             self.layers[f'Affine{i}'] = SimpleAffine(W,b)
#             self.layers[f'Sigmoid{i}'] = Sigmoid()
#         self.layers[f'Affine{i}'] = SimpleAffine(list(self.params.values())[-1][0],list(self.params.values())[-1][1])

#         self.lastLayer = SoftmaxWithLoss()
        

#     def predict(self, x):
#         for layer in self.layers.values():
#             x = layer.forward(x)
        
#         return x
    
#     def loss(self, x, t):
#         y = self.predict(x)
#         #return self.lastLayer.forward(y,t)

#     def accuracy(self, x, t):
#         pass

#     def gradient(self, x, t):
#         self.loss(x, t)

#         dout = 1
#         #dout = self.lastLayer.backward(dout)
#         for layer in reversed(self.layers.values()):
#             dout = layer.backward(dout)
        
#         grads = {}
#         for i, layer in enumerate(self.layers.values(),1):
#             if isinstance(layer, SimpleAffine):
#                 grads[f'W{i}'], grads[f'b{i}'] = layer.dW, layer.db
#         return grads
        

# network = NLayersNet(100,100,[100]*9,0.1)
# x = np.random.rand(1,100)
# grads = network.gradient(x,None)
# print(grads)

from common.layers import Affine, Sigmoid, Relu

layers=[]

b = np.zeros(100)

for k in range(11):
    W = 0.1*np.random.randn(100,100)
    layers.append(Affine(W,b))
    if k != 10:
        layers.append(Relu())

x = np.random.rand(1,100)

for k in range(21):
    x = layers[k].forward(x)

dout = np.random.rand(1,100)

for k in range(21):
    dout = layers[20-k].backward(dout)
    if k % 2 == 0:
        print(np.sum((layers[20-k].dW)**2))