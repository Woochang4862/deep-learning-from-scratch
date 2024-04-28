# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    if x.ndim == 2: # 행렬 : 배치처리 했을때 
        x = x.T # 전치 하는 이유 : np.max 가 axis에 상관없이 1xn을 반환하기 때문에
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))

def my_softmax(x):
    if x.ndim == 2: # 행렬 : 배치처리 했을때 
        x = x - np.array([np.max(x, axis=1)]).T # n np.array.T.shape == (n,)
        y = np.exp(x) / np.array([np.sum(np.exp(x), axis=1)]).T
        return y

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t): # 피타고라스
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t): # 정보이론
    if y.ndim == 1: # 벡터 : 배치처리 안 했을때 / batch_size 를 구하기 위해 행렬로 만들어줌
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size: # 원-핫 벡터 (batch_size)x(label_size) -> 원래 라벨 데이터 1x(batch_size) [0,2,1]
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    # y[np.array,np.array] == y[행번호들, 열번호들]
    # 1e-7 을 하는 이유 0에 너무 가까운 값이 들어가면 np.log의 값이 오버플로우가 나기때문에 
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size 


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
