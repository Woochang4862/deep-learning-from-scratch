# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle # 
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
import matplotlib.pyplot as plt
import time

np.set_printoptions(linewidth=150,threshold=1000)

def get_data(normalize=True,flatten=True,one_hot_label=False):
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=normalize, flatten=flatten, one_hot_label=one_hot_label)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
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

start_time = time.time()
x, t = get_data()
network = init_network()
y = []
list_of_is_correct = []
accuracy_cnt = 0
for i in range(len(x)):
    y_ = predict(network, x[i])
    p = np.argmax(y_) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    y.append(p)
    list_of_is_correct.append(p==t[i])
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
print(f"Excution Time : {time.time()-start_time}ms") # 0.9777185916900635 ms

# x,t = get_data(normalize=False,flatten=False)
# print(len(x))
# for i,v_img, label_true, label_predict, isCorrect in zip(range(len(x)),x,t,y,list_of_is_correct):
#     if isCorrect:
#         continue
#     print(v_img)
#     print(i,label_true,label_predict, isCorrect)
#     time.sleep(2)
# plt.figure()
# plt.imshow(img)
# plt.show()