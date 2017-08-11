import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.3)
config = tf.ConfigProto(gpu_options=gpu_options)

def read_and_decode(data_path): 
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
                batch_size=batch_size, capacity=300, num_threads=16,
                min_after_dequeue=10)
	
    return image_batch, label_batch


data_path = 'dataset/output/train.tfrecords'
batch_size = 10

imgs, labels = read_and_decode(data_path)

# Initialize all global and local variables
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session(config=config) as sess:
    sess.run(init_op)
	
	# Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
									   
    for batch_index in range(batch_size/3):		
        images, lbls = sess.run([imgs, labels])
        #images = images.astype(np.uint8)
        images_ = tf.reshape(images, [batch_size,256,256,3])
        print(images.shape)
        for j in range(6):        
            plt.subplot(2,3,j+1)
            plt.imshow(images[j])
            plt.title('bird' if lbls[j]==1 else 'fish')
        plt.show()
	
    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    sess.close()
									   