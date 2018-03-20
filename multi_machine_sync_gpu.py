from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from inception import inception_distributed_train
FLAGS = tf.app.flags.FLAGS


def main(unused_args):
  assert FLAGS.job_name in ['ps', 'worker'], 'job_name must be ps or worker'

  # Extract all the hostnames for the ps and worker jobs to construct the
  # cluster spec.
  ps_hosts = FLAGS.ps_hosts.split(',') # ps0.example.com:2222-->[ps0.example.com:2222]
  worker_hosts = FLAGS.worker_hosts.split(',')
  #'worker0.example.com:2222,worker1.example.com:2222' -->['worker0.example.com:2222,worker1.example.com:2222']
  tf.logging.info('PS hosts are: %s' % ps_hosts)
  tf.logging.info('Worker hosts are: %s' % worker_hosts)


  cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,'worker': worker_hosts})
  server = tf.train.Server({'ps': ps_hosts,'worker': worker_hosts},job_name=FLAGS.job_name,task_index=FLAGS.task_id)

  if FLAGS.job_name == 'ps':
    # `ps` jobs wait for incoming connections from the workers.
    server.join()
  else:
    # `worker` jobs will actually do the work.
    dataset = ImagenetData(subset=FLAGS.subset)
    assert dataset.data_files()
    # Only the chief checks for or creates train_dir.
    if FLAGS.task_id == 0:
      if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)
    inception_distributed_train.train(server.target, dataset, cluster_spec)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()