# linear SVM

import numpy as np
import pandas as pd
import  sklearn.svm
import seaborn as sns
import  scipy.io as sio
import matplotlib.pyplot as plt

mat = sio.loadmat('./data/ex6data1.mat')
print(mat.keys())