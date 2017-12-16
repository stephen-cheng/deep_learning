import math
import tempfile
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# parameters on console (flags) and default
flags = tf.app.flags
flags.DEFINE_string("data_dir", "mnist_data", "Directory for storing mnist data")
flags.DEFINE_integer("hidden_units", 100, 
                    "Number of units in the hidden layer of the NN")
flags.DEFINE_integer("train_steps", 10000,
                    "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_integer("learning_rate", 0.01, "learning rate")

# sync replicas and replicas to aggregate
flags.DEFINE_boolean("sync_replicas", False, 
                    "Use the sync_replicas (synchronized replicas) mode, "
                    "wherein the parameter updates from workers are "
                    "aggregated before applied to avoid stale gradients")
flags.DEFINE_integer("replicas_to_aggregate", None, 
                    "Number of replicas to aggregate before parameter "
                    "update is applied (For sync_replicas mode only; "
                    "default: num_workers)")

# parameter server and worker host, job and task
flags.DEFINE_string("ps_hosts", "10.167.175.140", 
                   "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "10.167.173.114, 10.167.173.115",
                   "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_integer("task_index", None,
                    "Worker task index, should be >= 0. task_index=0 is "
                    "the master worker task the performs the variable initialization")
FLAGS = flags.FLAGS
image_pixels = 28

# main function
def main(unused_argv):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    
    # parse ps and worker address
    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("Must specify an explicit 'job_name'")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit 'task_index'")
    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    
    # server to cluster
    num_workers = len(worker_spec)
    cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, 
                            task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()
        
    # deploy devices
    is_chief = (FLAGS.task_index == 0)
    worker_device = "/job:worker/task:%d/gpu:0" % FLAGS.task_index
    with tf.device(tf.train.replica_device_setter(worker_device=worker_device,
                                                 ps_device="/job:ps/cpu:0",
                                                 cluster=cluster)):
        global_step  = tf.Variable(0, name="global_step", trainable = False)
        
    # build mlp network
    hid_w = tf.Variable(tf.truncated_normal([image_pixels*image_pixels,
                FLAGS.hidden_units], stddev=1.0/image_pixels), name="hid_w")
    hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")
    sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10],
                stddev=1.0/math.sqrt(FLAGS.hidden_units)), name="sm_w")
    sm_b = tf.Variable(tf.zeros([10]), name="sm_b")
    
    x = tf.placeholder(tf.float32, [None, image_pixels*image_pixels])
    y_ = tf.placeholder(tf.float32, [None, 10])
    hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
    hid = tf.nn.relu(hid_lin)
    y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
    
    # sync replicas and optimizer
    if FLAGS.sync_replicas:
        if FLAGS.replicas_to_aggregate is None:
            replicas_to_aggregate = num_workers
        else:
            replicas_to_aggregate = FLAGS.replicas_to_aggregate
        opt = tf.train.SyncReplicasOptimizer(opt, 
                replicas_to_aggregate=replicas_to_aggregate,
                total_num_replicas=num_workers, replica_id=FLAGS.task_index,
                name="mnist_sync_replicas")
    train_step = opt.minimize(cross_entropy, global_step=global_step)
    
    # queue runner and global params init
    if FLAGS.sync_replicas and is_chief:
        chief_queue_runner = opt.get_chief_queue_runner()
        init_tokens_op = opt.get_init_tokens_op()
        
    # local params init and distributed train supervisor
    init_op = tf.global_variables_initializer()
    train_dir = tempfile.mkdtemp()
    sv = tf.train.Supervisor(is_chief=is_chief,
                            logdir=train_dir,
                            init_op=init_op,
                            recovery_wait_secs=1,
                            global_step=global_step)
    
    # session params config
    sess_config = tf.ConfigProto(allow_soft_placement=True, 
                    log_device_placement=False,
                    device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])
    if is_chief:
        print("Worker %d: Initializing session..." % FLAGS.task_index)
    else:
        print("Worker %d: Waiting for session to be initialized..." % FLAGS.task_index)
    sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
    print("Worker %d: Session initialization complete." % FLAGS.task_index)
    
    # sync pattern
    if FLAGS.sync_replicas and is_chief:
        print("Starting chief queue runner and running init_tokens_op")
        sv.start_queue_runners(sess, [chief_queue_runner])
        sess.run(init_kokens_op)
        
    # train
    time_begin = time.time()
    print("Training begins @ %f" % time_begin)
    local_step = 0
    while True:
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        train_feed = {x: batch_xs, y_: batch_ys}
        
        _, step = sess.run([train_step, global_step], feed_dict=train_feed)
        local_step += 1
        
        now = time.time()
        print("%f: worker %d: training step %d done (global step: %d)" %
             (now, FLAGS.task_index, local_step, step))
        
        if step >= FLAGS.train_steps:
            break
            
    # training time and loss results
    time_end = time.time()
    print("Training ends @ %f" % time_end)
    training_time = time_end - time_begin
    print("Training elapsed time: %f s" % training_time)
    
    val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    val_xent = sess.run(cross_entropy, feed_dict=val_feed)
    print("After %d training step(s), validation cross entropy = %g" %
         (FLAGS.train_steps, val_xent))
	
# run
if __name__ == "__main__":
    tf.app.run()
	

