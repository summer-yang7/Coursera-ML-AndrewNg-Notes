# linear SVM
import numpy as np
import pandas as pd
import  sklearn.svm
import seaborn as sns
import  scipy.io as sio
import matplotlib.pyplot as plt

mat = sio.loadmat('./data/ex6data1.mat')
print(mat.keys())
data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data['y'] = mat.get('y')

data.head()

## visualize data
fig, ax = plt.subplots(figsize = (8, 6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['y'], cmap='Reds')

im