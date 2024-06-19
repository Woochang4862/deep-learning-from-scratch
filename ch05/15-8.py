import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False)

np.set_printoptions(threshold=1000,linewidth=2000)

for k in [1,2,3]:
    print("="*10+"평행이동시킨 픽셀 : "+str(k)+"="*10)
    x=np.zeros(x_test.shape)
    print(x.shape)

    x[:,:,:28-k,k:]=x_test[:,:,k:,:28-k]

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(t_test[i])
        plt.imshow(x[i][0], cmap=plt.cm.binary)
    plt.show()