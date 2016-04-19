
_Log = open('../input/adult.test','r');
_file = open('../input/test.csv','w');
for line in _Log:
	row = line.split(',')
	label = row[-1]
	if( len(label) < 2):
		break
	if label[1] == '<':
		row[-1]='0\n'
	else:
		row[-1]='1\n'
	
	for i in range(0,len(row)-1):
		_file.write(row[i]+",")
	_file.write(row[-1])
