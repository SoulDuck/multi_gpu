import tensorflow as tf

cluster  = tf.train.ClusterSpec({"worker":["192.168.0.16:2222"] , "ps" : ["192.168.0.4:2222"]})
server = tf.train.Server(cluster , job_name= 'ps' , task_index=0)
server.join()