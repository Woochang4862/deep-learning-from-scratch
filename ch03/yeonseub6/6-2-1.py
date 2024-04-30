import numpy as np


"""
y 를 행렬로 변환 
t 를 원핫인코딩 행렬로 변환
@param:
    y : predict prob np.array (vector, matrix) -> matrix
    t : true label  np.array (scalar, one-hot encoding) -> one-hot
@return:
    0.5 * np.sum((y-t)**2) / batch_size
"""
def mean_squared_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]

    if t.size != y.size: # 정수형 라벨
        one_hot = np.zeros(y.shape)
        one_hot[np.arange(batch_size), t] = 1
        t = one_hot

    return 0.5 * np.sum((y-t)**2) / batch_size

y=np.array([1,1,1]) # 싱글 데이터
t=np.array(0) # 정수형 
print(mean_squared_error(y,t))
y = np.array([1,1,1]) # 싱글 데이터
t = np.array([0,0,1]) # 원-핫
print(mean_squared_error(y,t))
y = np.zeros((10,10)) # 배치 데이터
y[:,:5] = 1/5
t = np.arange(10) # 정수형 라벨
print(mean_squared_error(y, t))
y = np.array([[1,1,1],[1,1,1]]) # 배치 데이터
t = np.array([1,1]) # 원-핫
print(mean_squared_error(y,t))