# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
import time
from tqdm import tqdm
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

batch_sizes = range(1,31)
exec_times = []
for batch_size in tqdm(batch_sizes):
    start_time = time.time()
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0

    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1) # 0 : "행" 사이의 이동 / 1 : "열" 사이의 이동
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    exec_times.append(time.time()-start_time)
max_len = max(list(map(lambda x : len(str(x)), exec_times)))
print('Executation Times:')
print(f' Batch Size |{"Excution Time".center(max_len+4)}')
for i, exec_time in enumerate(exec_times):
    print(f'{str(i+1).ljust(len(" Batch Size "))}|{(str(exec_time)+"ms").center(max_len+4)}')
print(f'Mean of Execution Times: {sum(exec_times)/len(exec_times)}ms')

plt.figure()
plt.plot(batch_sizes,exec_times)
plt.show()