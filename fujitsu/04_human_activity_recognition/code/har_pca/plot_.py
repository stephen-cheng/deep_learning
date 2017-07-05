import numpy as np
import matplotlib.pyplot as plt

def plot_(feature, target, feature_test, target_test, title_name, clf, clf_score):
	# preprocess dataset
	h = .02  # step size in the mesh
	X = np.concatenate((feature, feature_test), axis=0)
	x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	
	# Plot the decision boundary. For that, we will assign a color to each
	# point in the mesh [x_min, x_max]x[y_min, y_max].
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	ax = plt.subplot(111)
	plt.subplots_adjust(wspace=0.4, hspace=0.4)
	cm = plt.cm.coolwarm
	ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

	# Plot also the training points and testing points
	ax.scatter(feature[:, 0], feature[:, 1], c=target, cmap=cm,
			   edgecolors='k')
	ax.scatter(feature_test[:, 0], feature_test[:, 1], c=target_test, cmap=cm,
			   edgecolors='k', alpha=0.6)
	ax.set_xlim(xx.min(), xx.max())
	ax.set_ylim(yy.min(), yy.max())
	ax.set_xticks(())
	ax.set_yticks(())
	ax.set_title(title_name)
	ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % clf_score).lstrip('0'),
			size=15, horizontalalignment='right')
	plt.tight_layout()
	plt.savefig("%s.png" % title_name) 
	plt.show()
	
	
