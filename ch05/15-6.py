import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
import matplotlib.pyplot as plt

with open('../fashion_mnist.pkl', 'rb') as f:
    fashion_mnist = pickle.load(f)
print(fashion_mnist)

print(fashion_mnist.keys())
for value in fashion_mnist.values():
    print(value.shape)

x_train = fashion_mnist['x_train']
t_train = fashion_mnist['t_train']
x_test = fashion_mnist['x_test']
t_test = fashion_mnist['t_test']

np.set_printoptions(linewidth=150,threshold=1000)
print(x_train[0])

plt.figure(figsize=(20,20))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(t_train[i])
    plt.imshow(x_train[i], cmap=plt.cm.binary) # plt.cm.binary 흰색:0, plt.cm.gray 검정:0

plt.show()

x_train = x_train / 255
x_test = x_test / 255

from two_layer_net import TwoLayerNet

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]
batch_size = 100   # 미니배치 크기
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

confusion = np.zeros((10,10), dtype=int)

for k in range(len(x_test)):
    i=int(t_test[k])
    y = network.predict(x_test[k].reshape(1,28,28))
    j= np.argmax(y)
    confusion[i][j] += 1
    
print(confusion)

class_names=['Tshirt/top','Troser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

for i in range(10):
    precision = confusion[i,i]/np.sum(confusion[:,i])
    recall = confusion[i,i]/np.sum(confusion[i,:])ㅁ
    F1_score = 2*(precision*recall) / (precision+recall)
    print(class_names[i]+" : "+str(precision.round(2))+",  "+str(recall.round(2))+",  "+str(F1_score.round(2)))