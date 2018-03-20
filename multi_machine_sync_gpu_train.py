import tensorflow as tf
import time
import datetime
import os
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')
tf.app.flags.DEFINE_string('subset', 'train', 'Either "train" or "validation".')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')

# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in sync_replicas_optimizer.
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

# More details can be found in the sync_replicas_optimizer class:
# tensorflow/python/training/sync_replicas_optimizer.py
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")
tf.app.flags.DEFINE_integer('save_interval_secs', 10 * 60,
                            'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 180,
                            'Save summaries interval seconds.')

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# Learning rate decay factor selected from https://arxiv.org/abs/1604.00981
tf.app.flags.DEFINE_float('initial_learning_rate', 0.045,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94,
                          'Learning rate decay factor.')


RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0


def train(target , dataset , cluster_spec):

    num_workers = len(cluster_spec.as_dict()['worker'])
    num_parameter_servers = len(cluster_spec.as_dict()['ps'])

    if FLAGS.num_replicas_to_aggregate == 01:
        num_replicas_to_aggregate = num_workers
    else:
        num_replicas_to_aggregate = FLAGS.num_replicas_to_aggregate

    # Both should be greater than 0 in a distributd training
    assert num_workers > 0 and num_parameter_servers >0 , ('num_workers and num_parameter_servers must be >0')
    is_chief = (FLAGS.task_id ==0)

    # Ops are assigned to worker by default

    with tf.device('/job/worker/task:%d' % FLAGS.task_id):
        #Variables and its related init/assign ops are assigned to s
        with slim.scipes.arg_scope([slim.variables.variable ,slim.variables.global_step] ,
                                   device = slim.variables.VariableDeviceChooser(num_parameter_servers) ):
            global_step = slim.variables.global_step()

            # Calculate the learning rate Schedule.
            num_batches_per_epoch = (dataset.num_examples_per_epoch() / FLAGS.batch_size)
            decay_steps = int(num_batches_per_epoch * FLAGS.num_epoches_per_decay / num_replicas_to_aggregate)
            lr = tf.train.exponential_decay(FLAGS.initial_learning_rate, global_step, decay_steps,
                                            FLAGS.learning_rate_decay_factor, staircase=True)

            opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY, momentum=RMSPROP_MOMENTUM, epsilon=RMSPROP_EPSILON)
            num_batches_per_epoch = (dataset.num_examples_per_epoch()/ FLAGS.batch_size())
            logits
            loss
            losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
            losses += tf.get_collection(tf.Graphkeys.REGULARIZATION_LOSSES)
            total_loss = tf.add_n(losses , name='total_loss')
            if is_chief:
                loss_averages = tf.train.ExponentialMovingAverage(0.9 , name='avg')
                loss_averages_op = loss_averages.apply(losses + [total_loss])

                for l in losses + [total_loss]:
                    loss_name = l.op.name
                    tf.scalar_summary(loss_name + '(raw)',l)
                    tf.scalar_summary(loss_name  , loss_averages.average(l))

                with tf.control_dependencies([loss_averages_op]):
                    total_loss = tf.identity(total_loss)

                #Track the moving averages of all trainable variables
                # Note that we maintain a 'double-average' of the BatchNormalization
                #global statistics

                exp_moving_average = tf.train.ExponentialMovingAverage(inception.MOVING_AVERAGE_DECAY , global_step)


                variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())

                #Add histograms for model variables

                for var in variables_to_average:
                    tf.histogram_summary(var.op.name ,var)

                opt = tf.train.SyncReplicasOptimizer(opt , replicas_to_aggreate = num_replicas_to_aggregate ,
                                                     replica_id = FLAGS.task_id ,
                                                     total_num_replicas = num_workers ,
                                                     variables_to_average = exp_moving_average ,
                                                     variables_to_average = variables_to_average)

                batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)
                assert batchnorm_updates ,'Batchnorm updats are msising'
                batchnorm_updates_op = tf.group(*batchnorm_updates)

                with tf.control_dependencies([batchnorm_updates_op]):
                    total_loss = tf.identity(total_loss)
                #compute gradients with respect to loss
                grads = opt.compute_gradients(total_loss)

                for grad , var in grads:
                    if grad is not None:
                        tf.histogram_summary(var.op.name + '/gradients' , grad)
                apply_gradients_op = opt.apply_gradients(grads , global_step = global_step)

                with tf.control_dependencies([apply_gradients_op]):
                    train_op = tf.identity(total_loss , name='train_op')


                #get chief queue_runners , init_tokens and clean_up_op , which is used to synchronize replicas
                # More details can be found in sync_replicas_optimizer
                chief_queue_runners = [opt.get_chief_queue_runner()]
                init_tokens_op = opt.get_init_tokens_op()
                clean_up_op = opt.get_clean_up_op()

                saver = tf.train.Saver()
                summary_op = tf.merge_all_summaries()
                init_op = tf.initialize_all_variables()
                sv = tf.train.Supervisor(is_chief = is_chief ,
                                         logdir = FLAGS.train_dir,
                                         init_op = init_op,
                                         summary_op = None,
                                         global_step = global_step,
                                         saver = saver ,
                                         save_model_sces = FLAGS.save_interval_secs)
                tf.logging.info('%s Supervisor' %datetime.now())

                sess_config = tf.ConfigProto(allow_soft_placement =True ,
                                             log_device_placement = FLAGS.log_device_placement)

                # Get a session
                sess = sv.prepare_or_wait_for_session(target , config = sess_config)

                # Start the queue runners
                queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
                sv.start_queue_runners(sess ,queue_runners)
                tf.logging.into('Started %d queues for processing input data' , len(queue_runners))

                if is_chief:
                    sv.start_queue_runners((sess, chief_queue_runners))
                    sess.run(init_tokens_op)
                #Note that summary_op and train_op never run simultaneously in order to prevent running out of GPU memory

                next_summary_time = time.time() + FLAGS.save_summaries_secs
                while_not sv.should_stop():
                try:
                    start_time = time.time()
                    loss_value , step = sess.run([train_op , global_step])
                    assert_not np.isnan(loss_value) , 'Model diverged with loss = NaN'
                    if step > FLAGS.max_steps:
                        break;
                    duration = time.time() - start_time

                    if step %30 ==0:
                        examples_per_sec = FLAGS.batch_size / float(duration)
                        format_str = 'Worker %d: %s: step %d, loss = %.2f' '(%.1f examples/sec; %.3f sec_batch)')
                        tf.logging.info(format_str % (FLAGS.task_id , datetime.now() , step , loss_value , examples_per_sec , duratino))

                    if is_chief and next_summary_time < time.time():
                        tf.logging.info('Running Summary operation on the chief')
                        summary_str = sess.run(summary_op)
                        sv.summay_computed(sess, summary_str)
                        tf.logging.info('Finished running Summaries secs')
                except:
                    if is_chief:
                        tf.logging.info('About to execute sync_clean_up_op!')
                        sess.run(clean_up_op)
                    raise

                sv.stop()

                if is_chief:
                    saver.save(sess, os.path.join(FLAGS.train_dir , 'model.ckpt' , global_step= global_step))











