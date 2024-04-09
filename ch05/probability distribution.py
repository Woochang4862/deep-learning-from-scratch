# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
from common.functions import *

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

sample=x_test[0]

plt.figure()
plt.imshow(sample.reshape(28,28), cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.show()

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 1000
eval_interval=50
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

iter_per_epoch = max(train_size / batch_size, 1)

plt.figure(figsize=(10,10))
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(훨씬 빠르다)
    
    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    if (i % eval_interval == 0) & ((i//eval_interval)<16):
        probability=softmax(network.predict(sample.reshape(1,784)))
        print(probability)
        plt.subplot(4,4,int((i//eval_interval)+1))
        plt.bar(range(len(probability[0])),probability[0])
        plt.ylim(0, 1.0)
plt.show()
        
