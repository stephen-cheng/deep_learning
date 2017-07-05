import numpy as np

def write_data(filename, data):
	f = open(filename, 'w')	
	for line in data:
		for l in line:
			f.write(str(l)+'\t')
		f.write('\n')
	f.close()
	return filename

if __name__ == '__main__':
	filename = "train_pca.txt"
	filename = write_data(filename, data)
