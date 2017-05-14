# LSTM with Variable Length Input Sequences to One Character Output
import InsertTools
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from theano.tensor.shared_randomstreams import RandomStreams

def load_data():
	conn=InsertTools.getConnection()
	cur=conn.cursor()
	sql="select timestamp, logid, severity from log_s1;"
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
	dataset = datatrain[:100]
	# create mapping of strings to integers and the reverse
	char_to_int = dict((c, i) for i, c in enumerate(dataset))
	int_to_char = dict((i, c) for i, c in enumerate(dataset))
	# prepare the dataset of input to output pairs encoded as integers
	num_inputs = 100
	max_len = 5
	dataX = []
	dataY = []
	for i in range(num_inputs):
		start = numpy.random.randint(len(dataset)-2)
		end = numpy.random.randint(start, min(start+max_len,len(dataset)-1))
		sequence_in = dataset[start:end+1]
		sequence_out = dataset[end + 1]
		dataX.append([char_to_int[char] for char in sequence_in])
		dataY.append(char_to_int[sequence_out])
		print sequence_in, '->', sequence_out
	# convert list of lists to array and pad sequences if needed
	X = pad_sequences(dataX, maxlen=max_len, dtype='float32')
	# reshape X to be [samples, time steps, features]
	X = numpy.reshape(X, (X.shape[0], max_len, 1))
	# normalize
	X = X / float(len(dataset))
	# one hot encode the output variable
	y = np_utils.to_categorical(dataY)
	# create and fit the model
	model = Sequential()
	model.add(LSTM(32, input_shape=(X.shape[1], 1)))
	model.add(Dense(y.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'precision', 'recall'])
	model.fit(X, y, nb_epoch=100, batch_size=20, verbose=2, shuffle=False)
	# summarize performance of the model
	scores = model.evaluate(X, y, verbose=0)
	print("Model Accuracy: %.2f%%" % (scores[1]*100))
	# demonstrate some model predictions
	for i in range(20):
		pattern_index = numpy.random.randint(len(dataX))
		pattern = dataX[pattern_index]
		x = pad_sequences([pattern], maxlen=max_len, dtype='float32')
		x = numpy.reshape(x, (1, max_len, 1))
		x = x / float(len(dataset))
		prediction = model.predict(x, verbose=0)
		index = numpy.argmax(prediction)
		result = int_to_char[index]
		seq_in = [int_to_char[value] for value in pattern]
		print seq_in, "->", result