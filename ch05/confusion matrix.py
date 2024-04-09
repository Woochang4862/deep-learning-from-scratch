# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
import pickle
from dataset.mnist import load_mnist
from common.functions import softmax, relu

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("neuralnet.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2 = network['W1'], network['W2']
    b1, b2 = network['b1'], network['b2']

    a1 = np.dot(x, W1) + b1
    z1 = relu(a1)
    a2 = np.dot(z1, W2) + b2

    return a2


x, t = get_data()
network = init_network()
confusion = np.zeros((10,10), dtype=int)

for k in range(len(x)):
    i=int(t[k])
    y = predict(network, x[k])
    j= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    confusion[i][j] += 1
    
print(confusion)

accuracy=0

for k in range(len(confusion)):
    accuracy+=confusion[k][k]

accuracy=accuracy/np.sum(confusion)
print(accuracy)

number1=4
number2=9
number1_number2=[]
number2_number1=[]

for k in range(len(x)):
    i=int(t[k])
    y = predict(network, x[k])
    j= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if i==number1 and j==number2:
        number1_number2.append(k)
    if i==number2 and j==number1:
        number2_number1.append(k)

plt.figure(figsize=(10,10))
for i in range(len(number1_number2)):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x[number1_number2[i]].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(t[number1_number2[i]])
plt.show()

plt.figure(figsize=(10,10))
for i in range(len(number2_number1)):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x[number2_number1[i]].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(t[number2_number1[i]])
plt.show()