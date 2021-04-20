import pandas as pd
import seaborn as sns
sns.set(context="notebook",style="whitegrid", palette="dark")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

df = pd.read_csv('ex1data1.txt', names=['population', 'profit'])
# df.head()
# df.info()

sns.lmplot('population', 'profit', df, size=6, fit_reg=False)
plt.show()

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
                loss_data.append(loss_val[0, 0])