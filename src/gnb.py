import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

train_df = pd.read_csv('../input/train-red.csv', header=0)
test_df = pd.read_csv('../input/test-red.csv', header=0)


train_y = train_df['quality']
test_y = test_df['quality']

train_df.drop(['quality'],axis=1, inplace=True)
train_X = train_df.as_matrix()


test_df.drop(['quality'],axis=1, inplace=True)
test_X = test_df.as_matrix()


#gbm = xgb.XGBRegressor(max_depth=5, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
#gbm = linear_model.LinearRegression()
#gbm.fit(train_X,train_y)
#gnb = GaussianNB()
gnb = MultinomialNB()
gnb.fit(train_X,train_y)
predictions = gnb.predict(test_X)

_error = open("error.log",'w')
_error.write("actual, predicted, error\n")
for i in range(0,len(predictions)):
	_error.write(str(test_y[i])+","+str(predictions[i])+","+str(abs(float(predictions[i])-float(test_y[i])))+"\n")

_error.flush()
_error.close()
