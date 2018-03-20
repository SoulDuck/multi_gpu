import tensorflow as tf

cluster  = tf.train.ClusterSpec({"worker":["192.168.0.16:2222"] , "ps" : ["192.168.0.4:2222"]})
server = tf.train.Server(cluster , job_name= 'worker' , task_index=0)
#server.join() #only ps server using join ()method
print server.target
#assign ops to the local worker by default
with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:0" , cluster= cluster)):
    x_ = tf.placeholder(dtype=tf.float32, shape=[1], name='x_')
    y_ = tf.placeholder(dtype = tf.float32  ,shape=[1] ,name ='y_')
    a=tf.Variable(3)
    b=tf.Variable(2)
    c=a*b



