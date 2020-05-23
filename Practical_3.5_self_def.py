## 梯度下降法实现
import matplotlib.pyplot as plt
import numpy as np
# 1)实现 P59 公式3.27极大似然法

def likelihood_sub(x, y, beta):
    """
    :param x: 一个示例变量(行向量)
    :param y:一个样品标签(行向量)
    :param beta:3.27中矢量参数(行向量)
    :return: 单个对数似然 3.27
    """
    return -y * np.dot(beta, x.T) + np.math.log(1 + np.math.exp(np.dot(beta, x.T)))

def likelihood(X, y, beta):
    """
    公式 3.27 :对数似然函数(交叉熵损失函数)
    :param X: 示例变量矩阵
    :param y:样本标签矩阵
    :param beta:3.27中的矢量参数
    :return: beta 的似然值
    """
    sum = 0
    m, n = np.shape(X)

    for i in range(m):
        sum += likelihood_sub(X[i], y[i], beta)
    return sum

# 2)实现似然公式一阶偏导
def sigmoid(x, beta):
    """
    基础模型 S 形函数
    P59 对数几率回归(逻辑回归)公式 3.23
    :param x: 预测变量
    :param beta: beta 变量
    :return:S 形函数
    """
    return 1 / (1 + np.math.exp(- np.dot(beta, x.T)))
def partial_derivative(X, y, beta):
    """
    P60 似然公式一阶偏导3.30
    :param X:示例变量矩阵
    :param y:样本标签矩阵
    :param beta:3.27 中矢量参数
    :return: beta 的偏导数,梯度
    """
    m, n = np.shape(X)
    pd = np.zeros(n)

    for i in range(m):
        tmp = -y[i] + sigmoid(X[i], beta)
        for j in range(n):
            pd[j] += X[i][j] * tmp
    return pd

# 3) 批量梯度下降法
def gradDscent(X, y, alpha, iterations, n):
    """
    :param X:变量矩阵
    :param y:样本标签数组
    :return:3.27中beta参数最优解
    """
    cost = np.zeros(iterations) # 构建 max_times 个 0 的数组
    beta = np.mat(np.zeros(n)) # 初始化 beta

    for i in range(iterations):
        # 梯度下降
        output = partial_derivative(X, y, beta)
        beta = beta - alpha * output
        cost[i] = likelihood(X, y, beta)

    return beta, cost
# 4) 绘制收敛曲线
def showConvergCurve(Iterations, Cost):
    """
    :param Iterations: 迭代次数
    :param Cost: 损失值数组
    """
    f1 = plt.figure(1)
    t = np.arange(Iterations)
    p1 = plt.subplot(1,1,1)
    p1.plot(t, Cost, 'r')
    p1.set_xlabel('Iterations')
    p1.set_ylabel('cost')
    p1.set_title('The Gradient Descent Convergence Curve')

    plt.show()

# 5) 绘制决策边界
def showLogRegression(X, y, Beta, N):
    f2 = plt.figure(2)

    plt.title('The Logistic Regression Fitted Curve')
    plt.xlabel('density')
    plt.ylabel('rate_sugar')
    # f = Beta * X.transpose()
    # plt.plot(X[:, 2], f.tolist()[0], 'r', label = 'Prediction')
    min_x = min(X[:, 0])
    max_x = max(X[:, 0])
    y_min_x = (- Beta.tolist()[0][2] - Beta.tolist()[0][0] * min_x) / Beta.tolist()[0][1] # 由线性模型 y =  w1 * x1 + w2 * x2 +b
    y_max_x = (- Beta.tolist()[0][2] - Beta.tolist()[0][0] * max_x) / Beta.tolist()[0][1]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=100, label='bad')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=100, label='good')
    plt.legend(loc='upper right')
    plt.show()

# 6) 测试
def testLogRegres(Beta, test_x, test_y):
    m, n = np.shape(test_x)
    matchCount = 0
    for i in range(m):
        predict = sigmoid(test_x[i], Beta) > 0.5
        if predict == bool(test_y[i]):
            matchCount += 1
    accuracy = float(matchCount) / m
    return accuracy

def loadData():
    dataset = np.loadtxt('/home/data/watermelon_3a.csv', delimiter=',')

    X = dataset[:, 1: 3]
    tmp = np.ones(X.shape[0])
    X = np.insert(X, 2, values=tmp, axis=1) # 在最后一列插入全是 1 的列
    y = dataset[:, 3]
    return X, y

def main():
    alpha = 0.1  # 迭代步长
    iterations = 1500  # 迭代次数上限
    X, y = loadData()
    test_x = X
    test_y = y
    m, n = np.shape(X)
    beat, cost = gradDscent(X, y, alpha, iterations, n)
    showConvergCurve(iterations, cost)
    showLogRegression(X, y, beat, n)
    accuracy = testLogRegres(beat, test_x, test_y)
    print('The classify accuracy is: %.3f%%' %(accuracy * 100))

main()

