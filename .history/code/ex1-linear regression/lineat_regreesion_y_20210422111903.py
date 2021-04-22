import pandas as pd
import seaborn as sns
sns.set(context="notebook",style="whitegrid", palette="dark")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

df = pd.read_csv('ex1data1.txt', names=['population', 'profit'])
# df.head()
# df.info()

# sns.lmplot('population', 'profit', df, size=6, fit_reg=False)
# plt.show()

# 读取特征
def get_X(df):
    ones = pd.DataFrame({'ones': np.ones(len(df))}) #ones是m行一列的Dataframe
    data = pd.concat([ones, df], axis=1) #合并数据，根据列合并
    return data.iloc[:, :-1].values #这个操作返回ndarray,不是矩阵

# 读取标签
def get_y(df):
    return np.array(df.iloc[:, -1])

def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean())/column.std())

def linear_regression(X_data, y_data, alpha, epoch, optimizer=tf.train.GradientDescentOptimizer):
    X = tf.placeholder(tf.float32, shape=X_data.shape) #placeholder()函数是在神经网络构建graph的时候在模型中的占位
    y = tf.placeholder(tf.float32, shape=y_data.shape)

    # construct the graph
    with tf.variable_scope('regression'):
        W = tf.get_variable('weights', 
                            (X_data.shape[1], 1),
                            initializer=tf.constant_initializer()) # n*1
        y_pred = tf.matmul(X, W) # m*n @ n*1 -> m*1
        loss = 1 / (2 * len(X_data)) * tf.matmul((y_pred - y), (y_pred - y), transpose_a=True)  # (m*1).T @ m*1 = 1*1

        opt = optimizer(learning_rate=alpha)
        opt_operation = opt.minimize(loss)

        # run the session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            loss_data = []

            for i in range(epoch):
                _, loss_val, W_val = sess.run([opt_operation, loss, W], feed_dict={X: X_data, y: y_data})
                loss_data.append(loss_val[0, 0]) # because every loss_val is 1*1 ndarray

                if len(loss_data) > 1 and np.abs(loss_data[-1] - loss_data[-2]) < 10 ** -9:
                    break

        tf.reset_default_graph()
        return {'loss': loss_data, 'parameters': W_val} # just want to return in row vector format

data = pd.read_csv('ex1data1.txt', names=['pupulation', 'profit']) #读取数据，并赋予列名

# data.head()

# 计算代价函数

X = get_X(data)
y = get_y(data)

theta = np.zeros(X.shape[1])

def lr_cost(theta, x, y):
    """
    X: R(m*n), m 样本数, n 特征数
    y: R(m)
    theta : R(n), 线性回归的参数
    """  
    m =  X.shape[0] #m为样本数
    inner = X @ theta - y # R(m*1), X @ theta 等价于X.dot(theta)

    # 1*m @ m*1 = 1*1 in matrix multiplication
    # but you know numpy didn't do transpose in 1d array, so here is just a
    # vector inner product to itselves
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)

    return cost

lr_cost(theta, X, y)

def gradient(theta, X, y):
    m = X.shape[0]
    inner = X.T @ (X @ theta - y) # (m, n).T @ (m, 1) -> (n, 1), X @ theta等价于X.dot(theta)
    return inner / m

def batch_gradient_decent(theta, X, y, epoch, alpha=0.01):
    """
    拟合线性回归，返回参数和代价
    epoch: 批处理的轮数
    """
    cost_data = [lr_cost(theta, X, y)]
    _theta = theta.copy()

    for _ in range(epoch):
        _theta = _theta - alpha * gradient(_theta, X, y)
        cost_data.append(lr_cost(_theta, X, y))

    return _theta, cost_data

