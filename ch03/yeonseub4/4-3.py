import numpy as np
import matplotlib.pyplot as plt

X1 = np.arange(0,255+255/24,255/24).reshape(5,5)
X2 = X1.T
X3 = 255 - X1
X5 = np.array([0,0,255,0,0]*5).reshape(5,5)
X4 = X5.T
X6 = (X4+X5)/2
X = [X1, X2, X3, X4, X5, X6]
plt.figure(figsize=(3*2,2*2)) # 가로 세로
for i in range(len(X)):
    plt.subplot(2,3,i+1) # 세로 가로
    plt.xticks([]) # tick(눈금) 지우기
    plt.yticks([])
    plt.imshow(X[i], cmap=plt.cm.gray) # plt.cm.binary 0:흰색~1:검정색 / plt.cm.gray 0:검정색~1:흰색
plt.show()