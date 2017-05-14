import InsertTools
import sys
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
	
def load_data():
	conn=InsertTools.getConnection()
	cur=conn.cursor()
	sql="select timestamp, logid, severity from log_s1 limit 370000;"
	cur.execute(sql)
	res=cur.fetchall()
	sql2="select timestamp, logid, severity from log_p1 limit 30000;"
	cur.execute(sql2)
	res2=cur.fetchall()
	severity=['fatal','failure','critical','error','warning','notice','info','NULL']
	se_index=[0,1,2,3,4,5,6,7]
	se_raw,se_raw2,se_no,se_no2=[],[],[],[]
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
	return se_no,se_no2
	
if __name__=='__main__':
	#id=sys.argv[1]
	
	# fix random seed for reproducibility
	numpy.random.seed(7)

	# load the dataset
	dataset,dataset2 = load_data()
	dataset_all=dataset+dataset2
	dataset_all=numpy.asarray(dataset_all)
	dataset_all = dataset_all.astype('float32')

	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset_all = scaler.fit_transform(dataset_all)
	
	# split into train and test sets
	train_size = len(dataset)
	test_size = len(dataset_all) - train_size
	train, test = dataset_all[0:train_size,:], dataset_all[train_size:len(dataset_all),:]

	# reshape into X=t and Y=t+1
	look_back = 3
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)

	# reshape input to be [samples, time steps, features]
	trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
	testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(128, input_shape=(trainX.shape[1], trainX.shape[2])))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, nb_epoch=100, batch_size=100, verbose=2)

	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)

	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])

	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))

	# shift train predictions for plotting
	trainPredictPlot = numpy.empty_like(dataset_all)
	trainPredictPlot[:, :] = numpy.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

	# shift test predictions for plotting
	testPredictPlot = numpy.empty_like(dataset_all)
	testPredictPlot[:, :] = numpy.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset_all)-1, :] = testPredict

	# plot baseline and predictions
	plt.plot(scaler.inverse_transform(dataset_all))
	plt.plot(trainPredictPlot)
	plt.plot(testPredictPlot)
	plt.savefig('plot/lstm_log_s1_p1_%s.png' % look_back)
	plt.show()