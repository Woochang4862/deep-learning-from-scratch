# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from gradient_2d import numerical_gradient

class SimpleNet:
    def __init__(self, input_size, output_size, weight_init_std):
        self.W = np.random.randn(input_size,output_size)
        self.b = np.zeros(output_size)

    def predict(self, x):
        y = np.dot(x,self.W) + self.b
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        loss = 0.5*np.mean((y-t)**2)
        return loss

train_x, train_t = np.array([[1,1],[1,0],[0,1],[0,0]]), np.array([[1],[0],[0],[0]]) # bias shape : 4x1 = 4x2 * 2x1

network = SimpleNet(2,1,1)
learning_rate = 1
threshold = 0.5
loss = lambda W : network.loss(train_x,train_t)
plt.figure(figsize=(8,16)) # 가로세로 * 4
for i in range(8):
    dW = numerical_gradient(loss,network.W)
    network.W -= learning_rate*dW
    db = numerical_gradient(loss,network.b)
    network.b -= learning_rate*db

    plt.subplot(2,4,i+1)
    x1 = np.array([-2,2])
    x2 = - (network.W[0]/network.W[1]) * x1 + (threshold - network.b) / network.W[1]
    plt.plot(x1,x2,'r')
    pred = network.predict(train_x)
    mask_true = pred.flatten() > threshold # pred : 출력값의 shape은 4x1
    mask_false = pred.flatten() <= threshold 
    plt.plot(train_x[mask_true,0],train_x[mask_true,1], 'bo')
    plt.plot(train_x[mask_false,0],train_x[mask_false,1], 'kx')
    plt.xlim(-1,2)
    plt.ylim(-1,2)

plt.show()