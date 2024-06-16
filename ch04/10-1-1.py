# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import time

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:100]
t_batch = t_train[:100]

start_time = time.time()
grad = network.gradient(x_batch, t_batch)
end_time = time.time()
back_propagation_exec_time = end_time - start_time
print(f'back_propagation_exec_time : {back_propagation_exec_time}s')

start_time = time.time()
grad = network.numerical_gradient(x_batch, t_batch)
end_time = time.time()
numerical_gradient_exec_time = end_time - start_time
print(f'numerical_gradient_exec_time : {numerical_gradient_exec_time}s')
