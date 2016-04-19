import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as pp

train_df = pd.read_csv('../input/train.csv', header=0)

train_y = train_df['occupation']

feat_m = []
feat_l = []
for i in range(0,len(train_y)):
	if int(train_y[i]==0):
		feat_l.append(train_df['age'][i])
	else:
		feat_m.append(train_df['age'][i])
pp.plot(feat_m, "bx")
pp.plot(feat_l, "ro")
pp.show()
