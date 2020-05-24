## sklearn 逻辑斯蒂回归函数库实现

# 1) 载入数据,预处理
import numpy as np
import matplotlib.pyplot as plt


dataset = np.loadtxt('/home/data/watermelon_3a.csv', delimiter=',')

X = dataset[:, 1 : 3]
y = dataset[:, 3]
print(np.shape(X))

# 散点图
f1 = plt.figure(1)
plt.title('watermelon_3a')
plt.xlabel('density')
plt.ylabel('rate_sugar')
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker = 'o', color = 'k', s = 100, label= 'bad')
plt.scatter(X[y == 1, 0], X[y ==1, 1], marker= 'o', color = 'g', s = 100, label = 'good')
plt.legend(loc = 'upper right')
# plt.show()

# 2)sklearn 逻辑回归库拟合
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pylab as pl

# 切分数据集:留出法 返回 划分好的训练集测试集样本和训练集测试集标签
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=0)

# 训练模型
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# 模型测试
y_pred = log_model.predict(X_test)

# 打印混淆矩阵和相关度量
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)

# 绘制决策边界
f2 = plt.figure(2)
h = 0.01
x0_min, x0_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
x1_min, x1_max = X[:, 1].min() - 0.1, X[:, 1].max() +0.1
x0, x1= np.meshgrid(np.arange(x0_min, x0_max, h), np.arange(x1_min, x1_max, h)) # 生成笛卡尔积坐标矩阵

z = log_model.predict(np.c_[x0.ravel(), x1.ravel()]) # c_ 按列合并, ravel 降成一维

z = z.reshape(x0.shape)
plt.contourf(x0, x1, z, cmap = pl.cm.Paired)# 等高线

plt.title('watermelon_3a')
plt.xlabel('density')
plt.ylabel('rate_sugar')
plt.scatter(X[y==0, 0], X[y==0, 1], marker='o', color='k', s=100, label='bad')
plt.scatter(X[y==1, 0], X[y==1, 1], marker='o', color='g', s=100, label='good')
plt.legend(loc='upper right')
plt.show()



