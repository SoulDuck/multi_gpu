import tensorflow as tf

cluster  = tf.train.ClusterSpec({"worker":["175.211.95.83:5556"] , "ps" : ["127.0.0.1:2222"]})
server = tf.train.Server(cluster , job_name= 'worker' , task_index=0)
server.join()