import tensorflow as tf

# build a graph
a = tf.constant(5.0)
b = tf.constant(3.0)
c = a * b
# launch the graph in a session
sess = tf.Session()
print sess.run(c)
sess.close()



sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
# We can just use 'c.eval()' without passing 'sess'
print c.eval()
sess.close()



a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
with tf.Session():
  # We can also use 'c.eval()' here.
  print c.eval()