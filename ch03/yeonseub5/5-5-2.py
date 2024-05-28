# coding: utf-8
import sys, os
sys.path.append(os.pardir+'/../')  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
print(sys.path)
import numpy as np
import pickle # 
from ...dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
import time
from tqdm import tqdm

np.set_printoptions(linewidth=150,threshold=1000)

def get_data(normalize=True,flatten=True,one_hot_label=False):
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=normalize, flatten=flatten, one_hot_label=one_hot_label)
    return x_test, t_test


def init_network():
    with open("../sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

stds = [0.1, 0.2, 0.5]
noises = [std*np.random.randn(784) for std in stds]
for i, noise in enumerate(noises):
    start_time = time.time()
    y = []
    accuracy_cnt = 0
    for j in tqdm(range(len(x))):
        y_ = predict(network, x[j]+noise)
        p = np.argmax(y_) # 확률이 가장 높은 원소의 인덱스를 얻는다.
        y.append(p)
        if p == t[j]:
            accuracy_cnt += 1

    print(f"Accuracy with noise from std {stds[i]} :" + str(float(accuracy_cnt) / len(x)), end='')
    print(f"(excution time : {time.time()-start_time}ms)") # 0.9777185916900635 ms
    print('='*100)