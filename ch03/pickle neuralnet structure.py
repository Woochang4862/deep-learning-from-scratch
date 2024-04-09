# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle

np.set_printoptions(linewidth=1000,threshold=100000)

with open("sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)


W1, W2, W3 = network['W1'], network['W2'], network['W3']
b1, b2, b3 = network['b1'], network['b2'], network['b3']

print(W1)

print(type(network))
print(network.keys())
print('W1 shape:' + str(W1.shape))
print('W2 shape:' + str(W2.shape))
print('W3 shape:' + str(W3.shape))
print('b1 shape:' + str(b1.shape))
print('b2 shape:' + str(b2.shape))
print('b3 shape:' + str(b3.shape))


