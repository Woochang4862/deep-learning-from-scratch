# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
from tqdm import tqdm

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

total_acc_list_per_std = []
for std in tqdm([1,10,0]):
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10,weight_init_std=std)

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # 기울기 계산
        # 발등이 가장 올라가는 방향(각 변수들의 gradient 방향을 구함)
        # 의인화하면 내가 모르는 거 찾아내기
        #grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
        grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(훨씬 빠르다)
        
        # 갱신
        # 의인화 : 모르는 거를 알게 되는 방향으로 학습
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
        
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
        if i % iter_per_epoch == 0: # 1 Epoch은 1회독 (데이터를 배치사이즈만큼 한번 쭉 보았을 때)
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            # print(train_acc, test_acc)
    total_acc_list_per_std.append((std, train_acc_list[-1], test_acc_list[-1]))
for std, train_acc, test_acc in total_acc_list_per_std:
    print(f'표준편차 {std}일 때 정확도 | train : {train_acc} / test : {test_acc}')
