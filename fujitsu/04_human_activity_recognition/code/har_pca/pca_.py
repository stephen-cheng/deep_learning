import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import load_data as ld
import write_data as wd

def pca_(raw_data, components_num):
	# plot pca
	#pca = PCA()
	pca = PCA(n_components=components_num)
	pca.fit(raw_data)
	exp_variance_ratio = pca.explained_variance_ratio_
	
	# decomposition
	#data_reduced = PCA(n_components = 100).fit_transform(raw_data)
	data_reduced = pca.fit_transform(raw_data)
	
	return data_reduced, exp_variance_ratio, pca

def pca_index(data, components_num):
	var_list, pca_index = [], []
	for i in range(len(data[0])):
		var_list.append(data[:,i].var())
	for i in sorted(var_list, reverse=True)[:components_num]:
		pca_index.append(var_list.index(i))
	
	'''
	data_pca = []
	for i in pca_index:
		data_ = (data[:,i].ravel())
		data_pca.append(data_)
	data_pca = np.transpose(np.array(data_pca))
	'''
	
	return pca_index

def pca_plot(pca):
	plt.figure(1, figsize=(4, 3))
	plt.clf()
	plt.axes([.2, .2, .7, .7])
	plt.plot(pca.explained_variance_, linewidth=2)
	plt.axis('tight')
	plt.xlabel('n_components')
	plt.ylabel('explained_variance_')
	plt.savefig('pca_plot.png')
	plt.show()
	
def pca_scatter(X, y):
	fig = plt.figure(1, figsize=(4, 3))
	plt.clf()
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
	plt.cla()
	for name, label in [('--1--', 1), ('--4--', 4), ('--6--', 6)]:
		ax.text3D(X[y == label, 0].mean(),
				  X[y == label, 1].mean() + 1.5,
				  X[y == label, 2].mean(), name,
				  horizontalalignment='center',
				  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
		
	# Reorder the labels to have colors matching the cluster results
	y = np.choose(y, [0, 4, 2, 3, 6, 5, 1]).astype(np.float)
	ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral, edgecolor='k')
	ax.w_xaxis.set_ticklabels([])
	ax.w_yaxis.set_ticklabels([])
	ax.w_zaxis.set_ticklabels([])
	plt.savefig('pca_scatter_3D.png')
	plt.show()

if __name__ == '__main__':
	data = ld.load_data('../../dataset/UCI HAR Dataset/train/X_train.txt', 'data')
	label = ld.load_data('../../dataset/UCI HAR Dataset/train/y_train.txt', 'label')
	components_num = 3
	data_reduced, exp_variance_ratio, pca = pca_(data, components_num)
	
	# store data reduced
	wd.write_data('train_pca.txt', data_reduced)
	
	# pca index
	pca_index = pca_index(data, components_num)
	print "pca feature index: ", pca_index
	
	print ('confidence interval: %s' % (str(exp_variance_ratio.sum())))
	print('explained variance ratio (first %d components): %s' % (components_num, str(exp_variance_ratio)))
	pca_plot(pca)

	pca_scatter(data_reduced, label.ravel())
	
	
	
