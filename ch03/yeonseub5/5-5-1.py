import sys, os
sys.path.append(os.pardir+'/../')  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

# normalize : 0~1 사이의 값으로 바꿔준다
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=True)

noises = [ std * np.random.randn(28,28) for std in [0.1, 0.2, 0.5] ]
for noise in noises:
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([]) # tick(눈금) 지우기
        plt.yticks([])
        plt.imshow(x_train[i][0] + noise, cmap=plt.cm.gray) # plt.cm.binary 0:흰색~1:검정색 / plt.cm.gray 0:검정색~1:흰색
        plt.xlabel(t_train[i])
plt.show()