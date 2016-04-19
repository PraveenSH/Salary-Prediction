import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans

train_df = pd.read_csv('../input/train-red.csv', header=0)
test_df = pd.read_csv('../input/test-red.csv', header=0)


train_y = train_df['quality']
test_y = test_df['quality']

train_df.drop(['quality'],axis=1, inplace=True)
train_X = train_df.as_matrix()


test_df.drop(['quality'],axis=1, inplace=True)
test_X = test_df.as_matrix()


k_means = KMeans(n_clusters=10, init='k-means++')
train_pred = k_means.fit_predict(test_X)
label_quality = {}
for i in range(0,len(train_pred)):
	label_quality[train_pred[i]] = train_y[i]
test_pred = k_means.predict(test_X)
predictions = []
for i in range(0,len(test_pred)):
	predictions.append(label_quality[test_pred[i]])
_error = open("error.log",'w')
_error.write("actual, predicted, error\n")
for i in range(0,len(predictions)):
	_error.write(str(test_y[i])+","+str(predictions[i])+","+str(abs(float(predictions[i])-float(test_y[i])))+"\n")

_error.flush()
_error.close()
