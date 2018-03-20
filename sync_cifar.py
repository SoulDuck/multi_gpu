#-*- coding:utf-8 -*-
from datetime import datetime
import os.path
import re
import time
import random
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cifar10
import glob
from cifar_input import *
import cifar_input



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('batch_size' , 64 , """Number of batch_size""")
tf.app.flags.DEFINE_boolean('use_fp16' , False , """use float16""")
tf.app.flags.DEFINE_string('summary_dir' , './summary' , """folder to save tensorflow summary""")
tf.app.flags.DEFINE_string('model_dir' , './model' , """folder to save tensorflow model""")

train_filenames = glob.glob('./cifar_10/cifar-10-batches-py/data_batch*')
test_filenames = glob.glob('./cifar_10/cifar-10-batches-py/test_batch*')
train_imgs, train_labs = get_images_labels(*train_filenames)
test_imgs, test_labs = get_images_labels(*test_filenames)

train_imgs=train_imgs/255.
test_imgs = test_imgs/255.


def next_batch(imgs, labs, batch_size):
    indices = random.sample(range(np.shape(imgs)[0]), batch_size)
    if not type(imgs).__module__ == np.__name__:  # check images type to numpy
        imgs = np.asarray(imgs)
    imgs = np.asarray(imgs)
    batch_xs = imgs[indices]
    batch_ys = labs[indices]
    return batch_xs, batch_ys

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var



def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(name,shape,tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var






def _activation_summary(x , TOWER_NAME):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))




def inference(images ,n_classes , tower_name):
    """Build the CIFAR-10 model.
    Args:
    images: Images returned from distorted_inputs() or inputs().
    Returns:
    Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
    gpu_num = tower_name.split('_')[-1] # tower_name = tower_0 or tower_1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',shape=[5, 5, 3, 64],stddev=5e-2,wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name='relu')
        _activation_summary(conv1 ,tower_name )
        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
    # conv2

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 64],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='relu')
        _activation_summary(conv2 , tower_name)
    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    top_conv = tf.identity(pool2 , 'top_conv')
    reshape=tf.contrib.layers.flatten(top_conv)


    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        dim=reshape.get_shape()[1]
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name='relu')
        _activation_summary(local3 , tower_name)


    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='relu')
        _activation_summary(local4 , tower_name)

      # linear layer(WX + b),
      # We don't apply softmax here because
      # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
      # and performs the softmax internally for efficiency.

    #softmax_linear
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, n_classes],
                                              stddev=1/192.0, wd=None)
        biases = _variable_on_cpu('biases', [n_classes],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax')
        _activation_summary(softmax_linear , tower_name)

        return softmax_linear


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def tower_loss(n_classes, images, labels, tower_name ):
    # tower_loss 는 scope 로 둘러 쌓여진다
    # 그래서 loss() 에서 Weight_Decay
    logits = inference(images , n_classes,tower_name)
    total_loss = loss(logits , labels) # 이미 collection에 넣어기 때문에 필요 없다.
    losses = tf.get_collection('losses' , tower_name)
    return losses , total_loss


def average_gradient(tower_grads):
    #tower_grads [(gradient0 , variable0) ,(gradient1 , variable1)...]
    #grads = []

    average_grads=[]
    for grad_and_vars in zip(*tower_grads):
        grads=[]
        for g, _ in grad_and_vars:
            expanded_g=tf.expand_dims(g ,0 )
            grads.append(expanded_g)

        grad = tf.concat(axis=0 , values=grads)
        grad = tf.reduce_mean(grad , 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad , v )
        average_grads.append(grad_and_var )
    return average_grads



def train():
    """
    IMAGE_SIZE = 24

    # Global constants describing the CIFAR-10 data set.
    NUM_CLASSES = 10
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
    NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

    # Constants describing the training process.
    MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
    NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
    LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
    INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.
    """
    with tf.Graph().as_default() , tf.device('/cpu:0'):
        #global step
        global_step = tf.get_variable( 'global_step',[],initializer=tf.constant_initializer(0) , trainable=False)
        num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)
        lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE, global_step, decay_steps,
                                        cifar10.LEARNING_RATE_DECAY_FACTOR, staircase=True)
        opt = tf.train.GradientDescentOptimizer(lr)
        # Get Images and labels for CIFAR-10.
        tower_grads = []
        xs_=[]
        ys_=[]
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' %i):
                    tower_name = '%s_%d' % ( 'tower' , i )
                    with tf.name_scope(tower_name) as tower_scope:

                        x_ = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='x_{}'.format(i))
                        y_ = tf.placeholder(dtype=tf.float32, shape=[None], name='y_{}'.format(i))
                        xs_.append(x_)
                        ys_.append(y_)
                        losses , total_loss= tower_loss(10, x_ , y_ , tower_name) # tower의 모든 losses을 return 한다
                        tf.get_variable_scope().reuse_variables()
                        #total_loss = tf.add_n(losses, name='total_loss')
                        # (Optional) for tracking loss
                        # attech a scalar summary to all indivisual losses and the total loss
                        #왜 타워 이름을 빼는거지 ??
                        for l in losses + [total_loss]:
                            loss_name = re.sub('%s_[0-9]*/' % tower_name, '', l.op.name) # tower 이름을 지운다
                            tf.summary.scalar(loss_name, l)
                        #왜 마지막 타워만 남겨두지
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES , tower_scope) #  해당 tower의 graph을 가져온다.
                        grads = opt.compute_gradients(total_loss) # 모든 gradient에 대한 loss 가 있다
                        tower_grads.append(grads)

        #sync point
        grads=average_gradient(tower_grads=tower_grads)
        summaries.append(tf.summary.scalar('learning_rate', lr))
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))


        print tf.global_variables()
        variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op = tf.group(apply_gradient_op , variables_averages_op)
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge(summaries)
        init=tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir , sess.graph)
        for step in range(FLAGS.max_steps):
            start_time=time.time()
            batch_xs_1, batch_ys_1 = next_batch(train_imgs, train_labs , FLAGS.batch_size)
            batch_xs_2, batch_ys_2 = next_batch(train_imgs, train_labs,FLAGS.batch_size)
            _ , loss_value = sess.run([train_op , total_loss ] , feed_dict= {xs_[0] : batch_xs_1, ys_[0]: batch_ys_1 ,
                                                                       xs_[1] : batch_xs_2, ys_[1]: batch_ys_2 })
            duration = time.time() - start_time

            if step % 10 ==0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ' 'sec/batch)')
                print (format_str %(datetime.now() , step , loss_value , examples_per_sec , sec_per_batch))

            if step % 100 ==0:
                summary_str = sess.run(summary_op , feed_dict= {xs_[0] : batch_xs_1, ys_[0]: batch_ys_1 ,
                                                                       xs_[1] : batch_xs_2, ys_[1]: batch_ys_2 })
                summary_writer.add_summary(summary_str , step)

            if step % 1000 ==0 or (step + 1 ) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.model_dir , 'model.ckpt')
                saver.save(sess , checkpoint_path , global_step = step )
    
def main(argv = None):
    train()

if __name__ == '__main__':
    tf.app.run()
