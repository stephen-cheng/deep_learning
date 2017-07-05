"""Human activity recognition using smartphones dataset and an LSTM RNN."""

# https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition

# Note that the dataset must be already downloaded for this script to work.
# To download the dataset, do:
#     $ cd data/
#     $ python download_dataset.py

import os
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics

#os.environ['CUDA_VISIBLE_DEVICES']=''

# Load "X" (the neural network's training and testing inputs)
def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file]])

        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))


# Load "y" (the neural network's training and testing outputs)
def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file]],
        dtype=np.int32)

    file.close()
    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1


class Config(object):
    """
    define a class to store parameters, 
    the input should be feature mat of training and testing
    Note: it would be more interesting to use a HyperOpt search space:
    https://github.com/hyperopt/hyperopt
    """
    def __init__(self, X_train, X_test):
        # Input data
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series

        # Training
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = self.train_count * 300
        self.batch_size = 1500
        self.display_iter = 30000  # To show test set accuracy during training

        # LSTM structure
        # Features count is of 9 (3*3D sensors features over time)
        self.n_inputs = len(X_train[0][0])
        self.n_hidden = 32  # nb of neurons inside the neural network
        self.n_classes = 6  # Final output classes
        
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))}
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random_normal([self.n_classes]))}


def LSTM_Network(_X, config):
    """Function returns a TensorFlow RNN with two stacked LSTM cells

    Two LSTM cells are stacked which adds deepness to the neural network.
    Note, some code of this notebook is inspired from an slightly different
    RNN architecture used on another dataset, some of the credits goes to
    "aymericdamien".

    Args:
        _X:     ndarray feature matrix, shape: [batch_size, time_steps, n_inputs]
        config: Config for the neural network.

    Returns:
        This is a description of what is returned.

    Raises:
        KeyError: Raises an exception.

      Args:
        feature_mat: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : matrix  output shape [batch_size,n_classes]
    """
    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, config.n_inputs])
    # new shape: (n_steps*batch_size, n_input)

    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, config.W['hidden']) + config.biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, config.n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, 
                  state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, 
                  state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], 
                  state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']


def extract_batch_size(_train, step, batch_size):   
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data.    
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 

    return batch_s


def one_hot(y_):
    """
    Function to encode output labels from number indexes.
    E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    """
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


if __name__ == "__main__":

    # -----------------------------
    # Step 1: load and prepare data
    # -----------------------------

    # Those are separate normalised input features for the neural network
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_"]

    # Output classes to learn how to classify
    LABELS = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"]

    DATA_PATH = "../../dataset/"
    DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
    print("\n" + "Dataset is now located at: " + DATASET_PATH)
    
    TRAIN = "train/"
    TEST = "test/"

    X_train_signals_paths = [DATASET_PATH + TRAIN + "Inertial Signals/" + signal \
                             + "train.txt" for signal in INPUT_SIGNAL_TYPES]
    X_test_signals_paths = [DATASET_PATH + TEST + "Inertial Signals/" + signal + \
                             "test.txt" for signal in INPUT_SIGNAL_TYPES]
    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)

    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    y_test_path = DATASET_PATH + TEST + "y_test.txt"
    
    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)

    # -----------------------------------
    # Step 2: define parameters for model
    # -----------------------------------

    config = Config(X_train, X_test)
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(X_test.shape, y_test.shape,
          np.mean(X_test), np.std(X_test))
    print("the dataset is therefore properly normalised, as expected.")

    # ------------------------------------------------------
    # Step 3: Let's get serious and build the neural network
    # ------------------------------------------------------

    x = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
    y = tf.placeholder(tf.float32, [None, config.n_classes])

    pred_y = LSTM_Network(x, config)

    # Loss,optimizer,evaluation
    l2 = config.lambda_loss_amount * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Softmax loss and L2
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred_y)) + l2
    optimizer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred_y, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    # --------------------------------------------
    # Step 4: Hooray, now train the neural network
    # --------------------------------------------
    
    # To keep track of training's performance
    test_losses = []
    test_accuracies = []
    train_losses = []
    train_accuracies = []
  
    # Launch the graph
    # Note that log_device_placement can be turned ON but will cause console spam with RNNs.
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)

    # Perform Training steps with "batch_size" amount of example data at each loop
    step = 1
    while step * config.batch_size <= config.training_epochs:
        batch_xs = extract_batch_size(X_train, step, config.batch_size)
        batch_ys = one_hot(extract_batch_size(y_train, step, config.batch_size))

        # Fit training using batch data
        _, loss, acc = sess.run([optimizer, cost, accuracy],
            feed_dict={x: batch_xs, y: batch_ys})

        train_losses.append(loss)
        train_accuracies.append(acc)  
    
        # Evaluate network only at some steps for faster training: 
        if (step*config.batch_size % config.display_iter == 0) or (step == 1) \
            or (step * config.batch_size > config.training_epochs):
        
            # To not spam console, show training accuracy/loss in this "if"
            print("Training epochs #" + str(step*config.batch_size) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}".format(acc))
        
            # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
            loss, acc = sess.run([cost, accuracy], feed_dict={x: X_test, y: one_hot(y_test)})
        
            test_losses.append(loss)
            test_accuracies.append(acc)

            print("Performance on test set: " + "Batch Loss = {}".format(loss) + \
                ", Accuracy = {}".format(acc))

        step += 1

    print("Optimization Finished!")

    # Accuracy for test data
    one_hot_predictions, accuracy, final_loss = sess.run([pred_y, accuracy, cost],
        feed_dict={x: X_test, y: one_hot(y_test)})

    test_losses.append(final_loss)
    test_accuracies.append(accuracy)

    print("Final result: " + "Batch Loss = {}".format(final_loss) + \
        ", Accuracy = {}".format(accuracy))


    '''
    best_accuracy = 0.0
    # Start training for each batch and loop epochs
    for i in range(config.training_epochs):
        for start, end in zip(range(0, config.train_count, config.batch_size),
                range(config.batch_size, config.train_count + 1, config.batch_size)):
            sess.run(optimizer, feed_dict={x: X_train[start:end],
                                           y: y_train[start:end]})

        # Test completely at every epoch: calculate accuracy
        pred_out, accuracy_out, loss_out = sess.run(
            [pred_y, accuracy, cost],
            feed_dict={
                x: X_test,
                y: y_test})

        print("traing iter: {},".format(i) +
              " test accuracy : {},".format(accuracy_out) +
              " loss : {}".format(loss_out))
        best_accuracy = max(best_accuracy, accuracy_out)

    print("")
    print("final test accuracy: {}".format(accuracy_out))
    print("best epoch's test accuracy: {}".format(best_accuracy))
    print("")
    '''
    # ------------------------------------------------------------------
    # Step 5: Training is good, but having visual insight is even better
    # ------------------------------------------------------------------
    
    font = {'family': 'Bitstream Vera Sans', 'weight': 'bold', 'size': 18}

    matplotlib.rc('font', **font)

    width = 12
    height = 12
    plt.figure(figsize=(width, height))

    indep_train_axis = np.array(range(config.batch_size, 
        (len(train_losses)+1)*config.batch_size, config.batch_size))
    plt.plot(indep_train_axis, np.array(train_losses), "b--", label="Train losses")
    plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

    indep_test_axis = np.append(np.array(range(config.batch_size, 
        len(test_losses)*config.display_iter, config.display_iter)[:-1]), [config.training_epochs])

    plt.plot(indep_test_axis, np.array(test_losses), "b-", label="Test losses")
    plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")

    plt.title("Training session's progress over iterations")
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('Training Progress (Loss or Accuracy values)')
    plt.xlabel('Training iteration')
    plt.savefig('har_lstm_loss_accuracy.png')
    plt.show()
    
    # ------------------------------------------------------------------
    # Step 6: And finally, the multi-class confusion matrix and metrics!
    # ------------------------------------------------------------------
    
    # Results
    predictions = one_hot_predictions.argmax(1)

    print("Testing Accuracy: {}%".format(100*accuracy))

    print("")
    print("Precision: {}%".format(100*metrics.precision_score(
        y_test, predictions, average="weighted")))
    print("Recall: {}%".format(100*metrics.recall_score(y_test, 
        predictions, average="weighted")))
    print("f1_score: {}%".format(100*metrics.f1_score(y_test, 
        predictions, average="weighted")))

    print("")
    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(y_test, predictions)
    print(confusion_matrix)
    normalised_confusion_matrix = np.array(confusion_matrix, 
        dtype=np.float32)/np.sum(confusion_matrix)*100

    print("")
    print("Confusion matrix (normalised to % of total test data):")
    print(normalised_confusion_matrix)
    print("Note: training and testing data is not equally distributed amongst classes, ")
    print("so it is normal that more than a 6th of the data is \
        correctly classifier in the last category.")

    # Plot: 
    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    plt.imshow(normalised_confusion_matrix, interpolation='nearest', cmap=plt.cm.rainbow)
    plt.title("Confusion matrix \n(normalised to % of total test data)")
    plt.colorbar()
    tick_marks = np.arange(config.n_classes)
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('har_lstm_confusion_matrix_heatmap.png')
    plt.show()

    
