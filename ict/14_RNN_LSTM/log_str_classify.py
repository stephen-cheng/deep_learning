# Multi-Class Classification
import InsertTools
import time  
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

def load_data():
	conn = InsertTools.getConnection()
	cur = conn.cursor()
	sql = "select timestamp, logid, severity from log_s1 limit 40000;"
	cur.execute(sql)
	res = cur.fetchall()
	sql2 = "select timestamp, logid, severity from log_p1 limit 100;"
	cur.execute(sql2)
	res2 = cur.fetchall()
	severity = ['fatal','failure','critical','error','warning','notice','info','NULL']
	se_index = [0,1,2,3,4,5,6,7]
	date_time,date_time2,timestamp,timestamp2,logid,logid2,se_raw,se_raw2,se_no,se_no2=[],[],[],[],[],[],[],[],[],[]
	for r in res:
		date_time.append(r[0])
		logid.append(r[1])
		se_raw.append(r[2])
	for sr in se_raw:
		number = se_index[severity.index(sr)]
		se_no.append([number])
	for r2 in res2:
		date_time2.append(r2[0])
		logid2.append(r2[1])
		se_raw2.append(r2[2])
	for sr2 in se_raw2:
		number2 = se_index[severity.index(sr2)]
		se_no2.append([number2])
	for dt in date_time:
		t = time.mktime(dt.timetuple())
		timestamp.append(t)
	for dt2 in date_time2:
		t2 = time.mktime(dt2.timetuple())
		timestamp2.append(t2)
	cur.close()
	conn.commit()
	conn.close
	return timestamp,timestamp2,logid,logid2,se_raw,se_raw2
	
if __name__=='__main__':

	# fix random seed for reproducibility
	seed = 7
	numpy.random.seed(seed)

	# load dataset
	timestamp,timestamp2,logid,logid2,se_raw,se_raw2 = load_data()
	dataset = []
	dataset.append(timestamp)
	dataset.append(logid)
	dataset.append(se_raw)
	dataset = numpy.array(dataset)
	dataset = numpy.transpose(dataset)
	X = dataset[:,0:2].astype(float)
	Y = dataset[:,2]

	# encode class values as integers
	encoder = LabelEncoder()
	encoder.fit(Y)
	encoded_Y = encoder.transform(Y)
	# convert integers to dummy variables (i.e. one hot encoded)
	dummy_y = np_utils.to_categorical(encoded_Y)

	from sklearn.cross_validation import train_test_split

	# Shuffle and split the dataset into the number of training and testing points above
	X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.1, random_state = 42, stratify=Y)

	# define The Neural Network baseline model 
	def baseline_model():
		# create model
		model = Sequential()
		model.add(Dense(4, input_dim=2, init='normal', activation='relu'))
		model.add(Dense(6, init='normal', activation='sigmoid'))
		# Compile model
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

	# Evaluate The Model with k-Fold Cross Validation
	estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
	kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
	results = cross_val_score(estimator, X_train, Y_train, cv=kfold)
	print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

	# Make Predictions
	estimator.fit(X_train, Y_train)
	print "Accuracy: {}%\n".format(estimator.score(X_test, Y_test) *100)

	predictions = estimator.predict(X_test)
	print(predictions)
	print(encoder.inverse_transform(predictions))
	