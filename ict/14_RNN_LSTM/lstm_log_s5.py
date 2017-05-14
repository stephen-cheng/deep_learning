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
	sql="select timestamp, logid, severity, count from log_s5 limit 200;"
	cur.execute(sql)
	res=cur.fetchall()
	severity=['error','failure','fatal','info','NULL','warning']
	se_index=[0,1,2,3,4,5]
	se_raw,se_no,count=[],[],[]
	for r in res:
		se_raw.append(r[2])
		count.append([r[3]])
	for sr in se_raw:
		number=se_index[severity.index(sr)]
		se_no.append([number])
	cur.close()
	conn.commit()
	conn.close
	return se_no,count
	
if __name__=='__main__':
	#id=sys.argv[1]
	
	# fix random seed for reproducibility
	numpy.random.seed(7)

	# load the dataset
	dataset,dataset2 = load_data()
	dataset2=numpy.asarray(dataset2)
	dataset = dataset2.astype('float32')

	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)

	# split into train and test sets
	train_size = int(len(dataset) * 0.8)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

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
	model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)

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
	trainPredictPlot = numpy.empty_like(dataset)
	trainPredictPlot[:, :] = numpy.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

	# shift test predictions for plotting
	testPredictPlot = numpy.empty_like(dataset)
	testPredictPlot[:, :] = numpy.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

	# plot baseline and predictions
	plt.plot(scaler.inverse_transform(dataset))
	plt.plot(trainPredictPlot)
	plt.plot(testPredictPlot)
	plt.savefig('plot/log_s5_lstm_%s.png' % look_back)
	plt.show()