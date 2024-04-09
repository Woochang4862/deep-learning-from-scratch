import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=False)

plt.figure()
plt.imshow(x_train[0][0])
plt.colorbar()
plt.show()


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i][0], cmap=plt.cm.binary)
    plt.xlabel(t_train[i])
plt.show()
