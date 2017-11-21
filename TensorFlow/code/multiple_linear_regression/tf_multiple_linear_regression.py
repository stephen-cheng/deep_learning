import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

# Generate our data
data_x = np.linspace(1, 8, 100)[:, np.newaxis]
data_y = np.polyval([1, -14, 59, -70], data_x) \
        + 1.5 * np.sin(data_x) + np.random.randn(100, 1)

# Add intercept data and normalize
model_order = 5
data_x = np.power(data_x, range(model_order))
data_x /= np.max(data_x, axis=0)

# Shuffle data and produce train and test sets
order = np.random.permutation(len(data_x))
portion = 20
test_x = data_x[order[:portion]]
test_y = data_y[order[:portion]]
train_x = data_x[order[portion:]]
train_y = data_y[order[portion:]]

# Create TensorFlow graph
init_param = lambda shape: tf.zeros(shape, dtype=tf.float32)

with tf.name_scope("IO"):
    inputs = tf.placeholder(tf.float32, [None, model_order], name="X")
    outputs = tf.placeholder(tf.float32, [None, 1], name="Yhat")

with tf.name_scope("LR"):
    W = tf.Variable(init_param([model_order, 1]), name="W")
    y = tf.matmul(inputs, W)
    
with tf.name_scope("train"):
    learning_rate = tf.Variable(0.5, trainable=False)
    cost_op = tf.reduce_mean(tf.pow(y-outputs, 2))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_op)

# Perform gradient descent to learn model
tolerance = 1e-3
# Perform Stochastic Gradient Descent
epochs = 1
last_cost = 0
alpha = 0.4
max_epochs = 50000

sess = tf.Session()
print "Beginning Training"
with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(tf.assign(learning_rate, alpha))
    writer = tf.summary.FileWriter("tboard", sess.graph) # Create TensorBoard files
    while True:

        sess.run(train_op, feed_dict={inputs: train_x, outputs: train_y})
            
        # Keep track of our performance
        if epochs%100==0:
            cost = sess.run(cost_op, feed_dict={inputs: train_x, outputs: train_y})
            print "Epoch: %d - Error: %.4f" %(epochs, cost)

            # Stopping Condition
            if abs(last_cost - cost) < tolerance or epochs > max_epochs:
                print "Converged."
                break
            last_cost = cost
            
        epochs += 1
    
    w = W.eval()
    print "w =", w
    print "Test Cost =", sess.run(cost_op, feed_dict={inputs: test_x, outputs: test_y})

# Plot the model obtained	
y_model = np.polyval(w[::-1], np.linspace(0,1,200))
plt.plot(np.linspace(0,1,200), y_model, c='g', label='Model')
plt.scatter(train_x[:,1], train_y, c='b', label='Train Set')
plt.scatter(test_x[:,1], test_y, c='r', label='Test Set')
plt.grid()
plt.legend(loc='upper left')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(0,1)
plt.savefig('tf_multiple_linear_regression.png')
plt.show()

