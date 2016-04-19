import matplotlib.pyplot as plt

_Log = open('error.log','r');
error = []
first_line = True;
total = float(0)
for line in _Log:
	if first_line:
		first_line = False
		continue
	row = line.split(',')
	diff = float(row[2].split('\n')[0]);
	total+=diff

print total
