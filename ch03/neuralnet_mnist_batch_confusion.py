# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
import time
import matplotlib.pyplot as plt

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
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
confusion = np.zeros((10,10), dtype = np.int64)

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1) # 0 : "행" 사이의 이동 / 1 : "열" 사이의 이동
    for _t, _p in zip(t[i:i+batch_size], p):
        confusion[_t][_p] += 1

print(f"Confusion Matrix : \n{confusion}")
print(f"Excution Time : {time.time()-start_time}ms") # 0.5445611476898193

col_sum = np.sum(confusion, axis=1)
row_sum = np.sum(confusion, axis=0)
print(f'(4라고 예측한 데이터 중에서 실제라벨이 4인 데이터 수) / (4라고 예측한 데이터 수) : {confusion[4][4]/col_sum[4]}')
print(f'(실제라벨이 4인 데이터 중에서 4라고 예측한 데이터 수) / (4라고 예측한 데이터 수) : {confusion[4][4]/row_sum[4]}')

plt.figure(figsize=(4,2))
plt.subplot(1,2,1)
plt.imshow(confusion, cmap=plt.cm.gray)
plt.subplot(1,2,2)
for i in range(10):    
    confusion[i][i] = 0
plt.imshow(confusion, cmap=plt.cm.gray)
plt.show()