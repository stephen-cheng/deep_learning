import tensorflow as tf

x = tf.Variable(1.0, name='x')
add_op = tf.add(x, tf.constant(1.5))
assign_op = tf.assign(x, add_op)
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	sess.run(assign_op)
	print(sess.run(x))