import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## 对于 iris 数据

# 1) 载入数据,查看分布情况
iris = sns.load_dataset('iris') # 在线载入自带的 iris 数据集
X = iris.values[:, 0 : 4]
y = iris.values[:, 4]

sns.set(style='white') # 风格设置
g = sns.pairplot(iris, hue='species', markers=['o', 's', 'D']) # 变量关系组图
plt.show()

# 2) sklearn 逻辑回归库
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import model_selection

log_model = LogisticRegression(max_iter=1000) # 增加最大迭代次数,也可以减少数据量
m, n = np.shape(X)

# 十折交叉验证 指定模型直接返回测试值
y_pred_10_fold = model_selection.cross_val_predict(log_model, X, y, cv=10)

# 打印精度
accuracy_10_fold = metrics.accuracy_score(y, y_pred_10_fold)
print('The accuracy of 10-fold cross-validation:', accuracy_10_fold)


# 留一法
accuracy_LOO = 0
# 计算 m 次测试的结果
for train_index, test_index in model_selection.LeaveOneOut().split(X):
    X_train, X_test = X[train_index], X[test_index] # 训练集样本,测试集样本
    y_train, y_test = y[train_index], y[test_index] # 训练集标签, 测试集标签
    log_model.fit(X_train, y_train) # 训练模型
    y_pred_LOO = log_model.predict(X_test) # 测试
    if y_pred_LOO == y_test:
        accuracy_LOO += 1
print('The accuracy of Leave-One-Out:', accuracy_LOO / m)


## 对于 transfusion 数据

# # 1) 载入数据,预处理
# transfusion = pd.read_csv('/home/data/transfusion.data')
# X = transfusion.values[:, 0 : 4]
# y = transfusion.values[:, 4]
#
# # sns.set(style='white') # 风格设置
# # g = sns.pairplot(transfusion, hue='whether he/she donated blood in March 2007', markers=['o', 's']) # 变量关系组图
# # plt.show()
#
# # 2) sklearn 逻辑回归库
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
# from sklearn import model_selection
#
# log_model = LogisticRegression(max_iter=1000) # 增加最大迭代次数,也可以减少数据量
# m, n = np.shape(X)
#
# # 十折交叉验证 指定模型直接返回测试值
# y_pred_10_fold = model_selection.cross_val_predict(log_model, X, y, cv=10)
#
# # 打印精度
# accuracy_10_fold = metrics.accuracy_score(y, y_pred_10_fold)
# print('The accuracy of 10-fold cross-validation:', accuracy_10_fold)
#
#
# # 留一法
# accuracy_LOO = 0
# # 计算 m 次测试的结果
# for train_index, test_index in model_selection.LeaveOneOut().split(X):
#     X_train, X_test = X[train_index], X[test_index] # 训练集样本,测试集样本
#     y_train, y_test = y[train_index], y[test_index] # 训练集标签, 测试集标签
#     log_model.fit(X_train, y_train) # 训练模型
#     y_pred_LOO = log_model.predict(X_test) # 测试
#     if y_pred_LOO == y_test:
#         accuracy_LOO += 1
# print('The accuracy of Leave-One-Out:', accuracy_LOO / m)