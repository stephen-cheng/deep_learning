# Stateful LSTM to learn one-char to one-char mapping
import InsertTools
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from sklearn.metrics import precision_recall_fscore_support

def load_data():
	conn=InsertTools.getConnection()
	cur=conn.cursor()
	sql="select timestamp, logid, severity from log_s1 limit 1000;"
	cur.execute(sql)
	res=cur.fetchall()
	sql2="select timestamp, logid, severity from log_p1;"
	cur.execute(sql2)
	res2=cur.fetchall()
	severity=['fatal','failure','critical','error','warning','notice','info','NULL']
	se_index=[0,1,2,3,4,5,6,7]
	logid_train,logid_test,se_raw,se_raw2,se_no,se_no2=[],[],[],[],[],[]
	for r in res:
		logid_train.append(r[1])
	for r in res2:
		logid_test.append(r[1])
	for r in res:
		se_raw.append(r[2])
	for sr in se_raw:
		number=se_index[severity.index(sr)]
		se_no.append([number])
	for r2 in res2:
		se_raw2.append(r2[2])
	for sr2 in se_raw2:
		number2=se_index[severity.index(sr2)]
		se_no2.append([number2])
	cur.close()
	conn.commit()
	conn.close
	return logid_train,logid_test
	
if __name__=='__main__':
	# fix random seed for reproducibility
	numpy.random.seed(7)
	# define the raw dataset
	datatrain, datatest = load_data()
	dataset = datatrain
	# create mapping of strings to integers and the reverse
	char_to_int = dict((c, i) for i, c in enumerate(dataset))
	int_to_char = dict((i, c) for i, c in enumerate(dataset))
	# prepare the dataset of input to output pairs encoded as integers
	seq_length = 1
	dataX = []
	dataY = []
	for i in range(0, len(dataset) - seq_length, 1):
		seq_in = dataset[i:i + seq_length]
		seq_out = dataset[i + seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])
		print seq_in, '->', seq_out
	# reshape X to be [samples, time steps, features]
	X = numpy.reshape(dataX, (len(dataX), seq_length, 1))
	# normalize
	X = X / float(len(dataset))
	# one hot encode the output variable
	y = np_utils.to_categorical(dataY)
	# create and fit the model
	batch_size = 1
	model = Sequential()
	model.add(LSTM(16, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(y.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'precision', 'recall'])
	for i in range(5):	
		model.fit(X, y, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
		model.reset_states()
	# summarize performance of the model
	scores = model.evaluate(X, y, batch_size=batch_size, verbose=0)
	print("Model Accuracy: %.2f%%" % (scores[1]*100))
	# demonstrate some model predictions
	x_true,x_pred = [],[]
	seed = [char_to_int[dataset[1]]]
	for i in range(0, len(dataset)-1):
		x = numpy.reshape(seed, (1, len(seed), 1))
		x = x / float(len(dataset))
		prediction = model.predict(x, verbose=0)
		index = numpy.argmax(prediction)
		x_pred.append(int_to_char[index])
		x_true.append(int_to_char[seed[0]])
		print int_to_char[seed[0]], "->", int_to_char[index]
		seed = [index]
	model.reset_states()
	y_true = numpy.array(x_true[1:])
	y_pred = numpy.array(x_pred[:-1])
	prfs = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=None)
	precision,recall,fscore,support = prfs[0],prfs[1],prfs[2],prfs[3]
	print "Model Precision: %.2f%%" % (precision*100) 
	print "Model Recall: %.2f%%" % (recall*100)
	print "Model Fscore: %.2f%%" % (fscore*100)
	# demonstrate a random starting point
	severity = dataset[8]
	seed = [char_to_int[severity]]
	print "New start: ", severity
	for i in range(0, 5):
		x = numpy.reshape(seed, (1, len(seed), 1))
		x = x / float(len(dataset))
		prediction = model.predict(x, verbose=0)
		index = numpy.argmax(prediction)
		print int_to_char[seed[0]], "->", int_to_char[index]
		seed = [index]
	model.reset_states()