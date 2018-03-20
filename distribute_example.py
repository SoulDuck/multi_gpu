import tensorflow as tf

c = tf.constant("hello  , distributed tensorflow ")
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)
print sess.run(c)