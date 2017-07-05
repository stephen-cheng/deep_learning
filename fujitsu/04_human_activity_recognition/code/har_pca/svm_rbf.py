'''
SVM-rbf classifier parameters optimization:
	
	C=1.0, 
	kernel='rbf', 
	degree=3, 
	gamma=2.0, 
	coef0=0.0, 
	shrinking=True, 
	probability=False, 
	tol=0.001, 
	cache_size=200, 
	class_weight=None, 
	verbose=False, 
	max_iter=-1
'''

import load_data as ld
from plot_ import plot_
from sklearn import svm

def svm_(feature, target, feature_test, target_test):
    # C: svm regularization parameter
    clf = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=2, 
		coef0=0.0, shrinking=True, probability=False, 
		tol=0.001, cache_size=200, class_weight=None, 
		verbose=False, max_iter=-1, decision_function_shape=None, 
		random_state=None)
    clf.fit(feature, target.ravel())
    clf_score = clf.score(feature_test, target_test)
    clf_predict = clf.predict(feature_test)
    return clf, clf_score
			
if __name__ == '__main__':
	
    # predict and evaluation 
    feature = ld.load_data("../../dataset/UCI HAR Dataset/train/X_train.txt", 'data')
    label = ld.load_data('../../dataset/UCI HAR Dataset/train/y_train.txt', 'label')
    feature_test = ld.load_data('../../dataset/UCI HAR Dataset/test/X_test.txt', 'data')
    label_test = ld.load_data('../../dataset/UCI HAR Dataset/test/y_test.txt', 'label')

    feature, feature_test = feature[:, :2], feature_test[:, :2]
    clf, clf_score = svm_(feature, label, feature_test, label_test)
    print 'svm-rbf prediction accuracy score: ', clf_score

    # plot 2D figure
    title_name = "SVM-RBF classifier"
    plot_(feature, label, feature_test, label_test, title_name, clf, clf_score)
	
	
	
	
	
	
