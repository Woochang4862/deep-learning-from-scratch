# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
from common.layers import softmax

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

sample = x_test[0]
iters_num = 1000
eval_interval=50
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

plt.figure(figsize=(10,10))
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    grad = network.gradient(x_batch, t_batch)
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    if i % eval_interval == 0 and i // eval_interval < 16:
        y = softmax(network.predict(sample.reshape(1,-1)))
        print(y.shape)
        plt.subplot(4,4,i//eval_interval+1)
        plt.bar(range(len(t_test[0])),y[0])
        plt.ylim(0,1)

plt.show()
