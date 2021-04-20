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