import pandas as pd
import numpy as np
import scipy as sc
import math
from collections import defaultdict
from heapq import nlargest
from operator import itemgetter

sz_x = 500
sz_y = 1000
def scale(x,y):
	dx = math.floor(sz_x*x/10)
	if dx < 0:
		dx = 0
	if dx >= sz_x:
	        dx = sz_x-1

	dy = math.floor(sz_y*y/10)
	if dy < 0:
		dy = 0
	if dy >= sz_y:
		dy = sz_y-1

	return dx, dy

###########  Data Loading ##################################################
train_df = open("../input/train_sample.csv", "r")
test_df = open("../input/test_sample.csv", "r")
train_df.readline()
test_df.readline()

########## Data cleaning ##################################################
cluster = defaultdict(lambda: defaultdict(int))
cluster_sorted = dict()

while True:
	line = train_df.readline().strip()
	if line == '':
		break
	row = line.split(',')
	X = float(row[1])
	Y = float(row[2])
	place_id = row[5]
	dx,dy = scale(X,Y)
	cluster[(dx,dy)][place_id] += 1

train_df.close()

for el in cluster:
        cluster_sorted[el] = nlargest(3, sorted(cluster[el].items()), key=itemgetter(1))

_submit = open("submit_sample.csv",'w')
_submit.write("row_id,place_id\n")
while True:
	line = test_df.readline().strip()
	if line == '':
		break
	row = line.split(',')
	row_id = row[0]
	X = float(row[1])
	Y = float(row[2])

	dx,dy = scale(X,Y)
	print str(X)+" "+str(Y)+"==>"+str(dx)+" "+str(dy)
	_submit.write(str(row_id)+",")
	seen = []
	tmp = (dx,dy)
	if tmp in cluster_sorted:
		frequent = cluster_sorted[tmp]
		for i in range(len(frequent)):
			if frequent[i][0] in seen:
				continue;
			if len(seen)==3:
				break;
			
			
			_submit.write(" "+str(frequent[i][0]))
			seen.append(frequent[i][0])
	_submit.write("\n")

_submit.flush()
_submit.close()
