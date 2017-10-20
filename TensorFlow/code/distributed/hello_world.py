import tensorflow as tf

# Start a TensorFlow server as a single-process "cluster".
c = tf.constant("Hello, distributed TensorFlow!")
server = tf.train.Server.create_local_server()
# Create a session on the server.
sess = tf.Session(server.target)  
print(sess.run(c))
