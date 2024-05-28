# coding: utf-8
import sys, os
sys.path.append(os.pardir+'/../')  # 부모 디렉터리의 파일을 가져올 수 있도록 
import numpy as np
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
import time

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    network = {}
    network['W1'] = np.random.randn(784,50)
    network['b1'] = np.zeros(50)
    network['W2'] = np.random.randn(50,100)
    network['b2'] = np.zeros(100)
    network['W3'] = np.random.randn(100,10)
    network['b3'] = np.zeros(10)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y

start_time = time.time()
x, t = get_data()
network = init_network()

batch_size = 100 # 배치 크기 : 크게 할수록, OOM 에러 발생 가능성 높아짐
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1) # 0 : "행" 사이의 이동 / 1 : "열" 사이의 이동
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

print(f"Excution Time : {time.time()-start_time}ms") # 0.5445611476898193