import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize, pos_tag
import nltk
import scipy as sc
from scipy.sparse import csr_matrix, vstack, hstack
#nltk.download('all')

stemmer = PorterStemmer()
wnl = WordNetLemmatizer()

###########  Data Loading ##################################################
train_df = pd.read_csv('../input/train.csv', encoding="ISO-8859-1", header=0)
test_df = pd.read_csv('../input/test.csv',  encoding="ISO-8859-1", header=0)
print "loading done"


########## Data cleaning ##################################################
train_o = train_df[:]
test_o = test_df[:]

train_y = train_df['salary']
test_y = test_df['salary']

train_df.drop(['age'],axis=1,inplace=True)
train_df.drop(['fnlwgt'],axis=1,inplace=True)
train_df.drop(['education_num'],axis=1,inplace=True)
train_df.drop(['capital_gain'],axis=1,inplace=True)
train_df.drop(['capital_loss'],axis=1,inplace=True)
train_df.drop(['hours_per_week'],axis=1,inplace=True)
train_df.drop(['salary'],axis=1, inplace=True)


test_df.drop(['age'],axis=1,inplace=True)
test_df.drop(['fnlwgt'],axis=1,inplace=True)
test_df.drop(['education_num'],axis=1,inplace=True)
test_df.drop(['capital_gain'],axis=1,inplace=True)
test_df.drop(['capital_loss'],axis=1,inplace=True)
test_df.drop(['hours_per_week'],axis=1,inplace=True)
test_df.drop(['salary'],axis=1, inplace=True)
print "cleaning done"

#####################  tfidf extraction ############################
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
train_X = train_df.as_matrix()

tX = []
for i in range(0,train_X.shape[0]):
	tmp = ""
	for j in range(0,train_X.shape[1]):
		s = str(train_X[i][j])
		s = s.replace("-"," ")
		tmp = tmp+" "+s
	
#	tmp =  ' '.join([stemmer.stem(z) for z in word_tokenize(tmp)])
	tX.append(tmp)
train_X = tfidf.fit_transform(tX)
train_X.todense()

test_X = test_df.as_matrix()
testX = []
for i in range(0,test_X.shape[0]):
	tmp = ""
	for j in range(0,test_X.shape[1]):
		s = str(test_X[i][j])
		s = s.replace("-"," ")	
		tmp = tmp+" "+s
#	tmp =  ' '.join([stemmer.stem(z) for z in word_tokenize(tmp)])
	testX.append(tmp)
test_X = tfidf.transform(testX)
test_X.todense()

print "feature extraction done"

###############     Learning and prediction        ############################################
age_train = train_o['age'].values
age_test = test_o['age'].values
hours_train = train_o['hours_per_week'].values
hours_test = test_o['hours_per_week'].values
cg_train = train_o['capital_gain'].values
cg_test = test_o['capital_gain'].values
cl_train = train_o['capital_loss'].values
cl_test = test_o['capital_loss'].values

age_train = csr_matrix(age_train).transpose()
age_test = csr_matrix(age_test).transpose()
hours_train = csr_matrix(hours_train).transpose()
hours_test = csr_matrix(hours_test).transpose()
cg_train = csr_matrix(cg_train).transpose()
cg_test = csr_matrix(cg_test).transpose()
cl_train = csr_matrix(cl_train).transpose()
cl_test = csr_matrix(cl_test).transpose()

print train_X.shape


train_X = hstack((train_X,age_train))
test_X = hstack((test_X,age_test))
train_X = hstack((train_X,hours_train))
test_X = hstack((test_X,hours_test))
train_X = hstack((train_X,cg_train))
test_X = hstack((test_X,cg_test))
train_X = hstack((train_X,cl_train))
test_X = hstack((test_X,cl_test))

	
print train_X.shape

gbm = xgb.XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
predictions = gbm.predict(test_X)

_error = open("error.log",'w')
_error.write("actual, predicted, error\n")
for i in range(0,len(predictions)):
	_error.write(str(test_y[i])+","+str(predictions[i])+","+str(abs(float(predictions[i])-float(test_y[i])))+"\n")

_error.flush()
_error.close()
