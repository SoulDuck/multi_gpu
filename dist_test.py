import tensorflow as tf
import argparse
import time
parser=argparse.ArgumentParser()

parser.add_argument('--job' , type=str)
parser.add_argument('--logs_path ' , type = str , default='./logs')
parser.add_argument('--n_epoch' ,  type = int , default = 10)
parser.add_argument('--batch_size' ,  type = int , default = 60)

args=parser.parse_args()
task_index=0

cluster  = tf.train.ClusterSpec({"worker":["192.168.0.16:2222"] , "ps" : ["192.168.0.4:2222"]})
server = tf.train.Server(cluster , job_name= 'worker' , task_index=task_index)

if args.job == 'ps':
    server.join()
else:
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:0" , cluster= cluster)):
        x_ = tf.placeholder(dtype=tf.float32, shape=[1], name='x_')
        y_ = tf.placeholder(dtype = tf.float32  ,shape=[1] ,name ='y_')
        lr = tf.placeholer(dtype = tf.float32 ,name='lr')
        a=tf.Variable(3)
        b=tf.Variable(2)
        train_op=a*b

        global_step = tf.get_variable('global_step' , []  , initializer=tf.constant_initializer(0) , trainable=False)
        with tf.name_scope('input'):
            x_ = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
            # target 10 output classes
            y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
            #
        with tf.name_scope('weights'):
            w1 = tf.Variable(tf.zeros([784 , 100]) , 'w1')
            w2 = tf.Variable(tf.zeros([100, 10]) , 'w2')

        with tf.name_scope('biases'):
            b1 = tf.Variable(tf.zeros([100]), 'b1')
            b2 = tf.Variable(tf.zeros([10]), 'b2')

        with tf.name_scope("softmax"):
            # y is our prediction
            z2 = tf.add(tf.matmul(x_, w1), b1)
            a2 = tf.nn.sigmoid(z2)
            z3 = tf.add(tf.matmul(a2, w2), b2)
            y = tf.nn.softmax(z3)

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(-tf.reduce_mean(y_ * tf.log(y) , reduction_indices=1))

        with tf.name_scope('train'):
            grad_op=tf.train.GradientDescentOptimizer(learning_rate=lr)

            """
            rep_op = tf.train.SyncReplicasOptimizer(
                grad_op,
                replicas_to_aggregate=len(workers),
                replica_id=FLAGS.task_index,
                total_num_replicas=len(workers),
                use_locking=True)
            rep_op.minimize(cross_entropy ,global_step=global_step)
            train_op.minimize(cross_entropy ,global_step=global_step)
            
            """
        with tf.name_scope('Accuracy'):
            # accuracy
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # create a summary for our cost and accuracy
        tf.summary.scalar("cost", cross_entropy)
        tf.summary.scalar("accuracy", accuracy)

        # merge all summaries into a single "operation" which we can execute in a session
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        print("Variables initialized ...")

        sv = tf.train.Supervisor(is_chief=(task_index == 0),global_step=global_step,init_op=init_op)
        begin_time = time.time()
        frequency = 100

        with sv.prepare_or_wait_for_session(server.target) as sess:
            writer = tf.summary.FileWriter(args.logs_path ,graph=tf.get_default_graph())
            epoch_time = time.time()
            for epoch in range(args.n_epoch):
                batch_count = int(mnist.train.num_examples/args.batch_size)
                count =0
                for i in range(batch_count):
                    batch_x, batch_y = mnist.train.next_batch(args.batch_size)
                    _, cost, summary, step = sess.run([train_op, cross_entropy, summary_op, global_step],
                                                      feed_dict={x_: batch_x, y_: batch_y})
                    writer.add_summary(summary, step)
                    count +=1
                    if epoch % frequency == 0 or i+1 == batch_count:
                        elapsed_time = time.time() - epoch_time()
                        epoch_time = time.time()
                        print("Step: %d," % (step + 1),
                              " Epoch: %2d," % (epoch + 1),
                              " Batch: %3d of %3d," % (i + 1, batch_count),
                              " Cost: %.4f," % cost,
                              " AvgTime: %3.2fms" % float(elapsed_time * 1000 / frequency))
                        count = 0
            print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
            print("Total Time: %3.2fs" % float(time.time() - begin_time))
            print("Final Cost: %.4f" % cost)
        sv.stop()
        print "done"