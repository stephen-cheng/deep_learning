from __future__ import division, print_function, absolute_import
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.6)
config = tf.ConfigProto(gpu_options=gpu_options)


def read_and_decode(data_path, batch_size): 
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
	
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=
              {'image/encoded': tf.FixedLenFeature([], tf.string),
               'image/class/label': tf.FixedLenFeature([], tf.int64)})

    # Convert the image data from string back to the numbers
    img = tf.decode_raw(features['image/encoded'], tf.uint8)

    # Reshape image data into the original shape
    image = tf.reshape(img, [256, 256, 3])
    #image = tf.cast(image, tf.uint8)
									   
    # Cast label data into int32
    label = tf.cast(features['image/class/label'], tf.int32)
	
    # Creates batches by randomly shuffling tensors
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                batch_size=batch_size, capacity=200, num_threads=32,
                min_after_dequeue=16)
	
    return image_batch, label_batch

if __name__ == '__main__':
	
    # load data
    data_train = 'dataset/output/train.tfrecords'
    data_test = 'dataset/output/validation.tfrecords'
    batch_train = 2000
    batch_test = 400
    imgs_train, labels_train = read_and_decode(data_train, batch_train)
    imgs_test, labels_test = read_and_decode(data_test, batch_test)
	
    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session(config=config) as sess:
        sess.run(init_op)
	    # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
									   		
        images_train, lbls_train = sess.run([imgs_train, labels_train])
        #images = images.astype(np.uint8)
        images_ = tf.reshape(images_train, [batch_train,256,256,3])
        print(images_train.shape)
		
        images_test, lbls_test = sess.run([imgs_test, labels_test])
        #images = images.astype(np.uint8)
        images_ = tf.reshape(images_test, [batch_test,256,256,3])
        print(images_test.shape)
            
        # Stop the threads
        coord.request_stop()
        # Wait for threads to stop
        coord.join(threads)
        sess.close()
		
        # ont hot
        X = images_train
        testX = images_test
        Y = tflearn.data_utils.to_categorical(lbls_train, 10)
        testY = tflearn.data_utils.to_categorical(lbls_test, 10)
        print(X.shape, testX.shape, Y.shape, testY.shape)
		
		
        # build alexnet network
        network = input_data(shape=[None, 256, 256, 3])
        network = conv_2d(network, 96, 11, strides=4, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, 256, 5, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 10, activation='softmax')
        network = regression(network, optimizer='momentum',
                  loss='categorical_crossentropy', learning_rate=0.001)
		
        # train
        if not os.path.isdir('checkpoints'):
            os.makedirs('checkpoints')
        if not os.path.isdir('model'):
            os.makedirs('model')
			
        model = tflearn.DNN(network, checkpoint_path='checkpoints/model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
        model.fit(X, Y, n_epoch=200, validation_set=(testX, testY), shuffle=True,
                  show_metric=True, batch_size=64, snapshot_step=200,
                  snapshot_epoch=False, run_id='alexnet')
        model.save('model/model_retrained_by_alexnet')
        			   