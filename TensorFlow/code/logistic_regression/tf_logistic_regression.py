import numpy as np
import pandas as pd

# parameter init
learning_rate = 0.001
n_epochs = 100


def sigmod(para_x, para_w):
	wx = np.array([np.dot(x,para_w) for x in para_x])
	return 1/(1+np.exp((-1)*wx))


def ave_loss(para_x, para_w, para_y):
	wx = np.array([np.dot(x,para_w) for x in para_x])
	tmp = np.exp(wx+1)
	return (-1)*np.average(para_y*wx-np.log(tmp))


def gradient(para_x, para_w, para_y):
	wx = np.array([np.dot(x,para_w) for x in para_x])
	tmp = para_y - 1/(1+np.exp((-1)*wx))
	return (-1)*sum([para_x[ii]*tmp[ii] for ii in range(len(tmp))])


'''
	data (x_1, x_2, 1)	
	label y				\in {0,1}
	weight (w_1, w_2, b)
'''
data = pd.read_csv('Labeled_Data_2CLS_100.CSV')
cls = data.values[:,2]
data = data.values[:,:2]
x_bias = np.ones(len(data))
data = np.column_stack((data, x_bias))
weight = np.random.random(size=3)

for i in range(n_epochs):
		print('epoch :', i)
		print('average loss :', ave_loss(data, weight, cls))
		weight -= gradient(data, weight, cls) * learning_rate

		
#tf		
		
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd


# read data
data = pd.read_csv('Labeled_Data_2CLS_100.CSV')
temp = data.values[:, 2]
ys = np.array([[1, 0] for x in temp if x == 0] + [[0, 1] for x in temp if x == 1])
xs = data.values[:, :2]

# parameter init
n_epochs = 100
eta = 0.01


n_input = 2
n_output = 2


X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_output])
W = tf.Variable(tf.zeros([n_input, n_output]))
b = tf.Variable(tf.zeros([n_output]))
y_hat = tf.nn.softmax(tf.matmul(X, W) + b)              # softmax function
cross_entropy = -tf.reduce_sum(Y * tf.log(y_hat))
correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
optimizer = tf.train.GradientDescentOptimizer(eta).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prev_training_cost = 0.0
    for i_epoch in range(n_epochs):
        for (x, y) in zip(xs, ys):
            # the 1-dim matrix in numpy is terrible!!
            # this is a solution I find in overstack.
            # overstack is a good place!
            temp = x.shape
            x = x.reshape(1, temp[0])
            temp = y.shape
            y = y.reshape(1, temp[0])
            sess.run(optimizer, feed_dict={X: x, Y: y})

        if i_epoch % 20 == 0:
            training_cost = sess.run(accuracy, feed_dict={X: xs, Y: ys})
            print(training_cost)
    print(sess.run(W), sess.run(b))