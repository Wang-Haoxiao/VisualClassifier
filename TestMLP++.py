import MLP2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# 使用 sklearn 的 fetch_openml 函数加载 MNIST 数据集
mnist = fetch_openml('mnist_784')
X, y = mnist["data"].to_numpy(), mnist["target"].to_numpy()
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# 将目标值转换为整数类型
y_train = y_train.astype(np.int)
y_test = y_test.astype(np.int)

# 定义神经网络的层结构
layers = [X_train.shape[1], 50, 50, 10]  # 784个输入特征，两个隐藏层（每层50个神经元），10个输出

# 初始化神经网络
nn = MLP2.NeuralNetMLP2(layers=layers,
                      l2=0.1,
                      l1=0.1,
                      epochs=100,
                      eta=0.001,
                      alpha=0.001,
                      decrease_const=0.00001,
                      minibatches=50,
                      shuffle=True,
                      random_state=1)

# 训练神经网络
nn.fit(X_train, y_train, print_progress=True)

# 绘制成本曲线
plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs * 50')
plt.tight_layout()
plt.show()

# 平均成本曲线
batches = np.array_split(range(len(nn.cost_)), 1000)
cost_ary = np.array(nn.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]
plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()

# 计算训练集和测试集的准确率
y_train_pred = nn.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))

y_test_pred = nn.predict(X_test)
acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Test accuracy: %.2f%%' % (acc * 100))

# 显示被错误分类的图像
miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
