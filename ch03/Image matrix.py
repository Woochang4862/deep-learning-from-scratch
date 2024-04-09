import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
import time

np.set_printoptions(linewidth=150,threshold=1000)

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=False)

for x, t in zip(x_train, t_train):
    print(x)
    print(t)
    time.sleep(2)
