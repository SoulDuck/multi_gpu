import tensorflow as tf
import time
lr = 10
batch_size = 60
n_epoch = 3
frequency = 100

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
with tf.name_scope('input'):
    x_ = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
    # target 10 output classes
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
    #
with tf.name_scope('weights'):
    w1 = tf.Variable(tf.zeros([784, 100]), 'w1')
    w2 = tf.Variable(tf.zeros([100, 10]), 'w2')

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
    cross_entropy = tf.reduce_mean(-tf.reduce_mean(y_ * tf.log(y), reduction_indices=1))

with tf.name_scope('train'):
    grad_op = tf.train.GradientDescentOptimizer(learning_rate=lr)
    train_op = grad_op.minimize(cross_entropy, global_step=global_step)

with tf.name_scope('Accuracy'):
    # accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
for epoch in range(n_epoch):
    batch_count = int(mnist.train.num_examples / batch_size )
    count = 0
    for i in range(batch_count):
        epoch_time=time.time()
        batch_x, batch_y = mnist.train.next_batch(batch_size )
        _, cost, step = sess.run([train_op, cross_entropy, global_step],feed_dict={x_: batch_x, y_: batch_y})
        print cost
        count += 1
        """
        if epoch % frequency == 0 or i + 1 == batch_count:
            elapsed_time = time.time() - epoch_time
            epoch_time = time.time()
            print("Step: %d," % (step + 1),
                  " Epoch: %2d," % (epoch + 1),
                  " Batch: %3d of %3d," % (i + 1, batch_count),
                  " Cost: %.4f," % cost,
                  " AvgTime: %3.2fms" % float(elapsed_time * 1000 / frequency))
            count = 0
        """
