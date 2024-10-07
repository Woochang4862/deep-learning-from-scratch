import numpy as np
import matplotlib.pyplot as plt

N=100
x = np.arange(0, 1, 1/N)
y = np.arange(0, 1, 1/N)
X, Y = np.meshgrid(x, y)
Z = (X**2 + Y**2 <1)

plt.figure(figsize=(10,10))
# 순서가 달라지면 0,0 에 점 찍힘! 즉 점 200개를 찍는 거임
plt.plot(X.flatten()*(1-Z.flatten()), Y.flatten()*(1-Z.flatten()), '.', color="k") 
plt.plot(X.flatten()*Z.flatten(), Y.flatten()*Z.flatten(), '.', color="r")
plt.show()