import tensorflow as tf

task_index=0
cluster  = tf.train.ClusterSpec({"worker":["192.168.0.16:2222"] , "ps" : ["192.168.0.4:2222"]})
server = tf.train.Server(cluster , job_name= 'worker' , task_index=task_index)

#server.join() #only ps server using join ()method
print server.target
#assign ops to the local worker by default
with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:0" , cluster= cluster)):
    x_ = tf.placeholder(dtype=tf.float32, shape=[1], name='x_')
    y_ = tf.placeholder(dtype = tf.float32  ,shape=[1] ,name ='y_')
    a=tf.Variable(3)
    b=tf.Variable(2)
    train_op=a*b


    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(task_index == 0),
                                           checkpoint_dir="/tmp/train_logs",
                                           hooks=hooks) as mon_sess:
      while not mon_sess.should_stop():
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.
        mon_sess.run(train_op)
