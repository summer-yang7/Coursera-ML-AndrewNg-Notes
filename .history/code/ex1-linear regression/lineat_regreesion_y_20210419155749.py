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

def get_X(df):
    ones = pd.DataFrame({'ones': np.ones(len(df))}) #ones是m行一列的Dataframe
    data = pd.concat(([ones, df], axis=1)) #合并数据，根据列合并
    return data.iloc[:,:-1].as_matrix() #这个操作返回ndarray,不是矩阵

a = get_X(df)