import tensorflow as tf
import re
import numpy as np
sess=tf.Session()

print 'aaaaa','bbbb'

with tf.name_scope('a') as scope:
    print scope
    a=tf.Variable(1)
    print a
    a_add = tf.add_to_collection('int', a)
    a_list = tf.get_collection('int' , scope)


with tf.name_scope('b') as scope:
    b=tf.Variable(2)
    b_add = tf.add_to_collection('int', b)
    b_list = tf.get_collection('int' , scope)

initializer = tf.truncated_normal_initializer(stddev=1, dtype=tf.float32)
c = tf.get_variable('c', [2,3], initializer=initializer, dtype=tf.float32)
c_l2_loss=tf.nn.l2_loss(c)

all=tf.Variable(3)
all_add = tf.add_to_collection('int', all)
all_list = tf.get_collection('int')


tmp_holder =tf.placeholder(dtype=tf.float32 , shape=[3] , name = 'tmp_placeholder')



init = tf.global_variables_initializer()
sess.run(init)
print sess.run(a_list)
print sess.run(b_list)
print sess.run(all_list)
print sess.run(c)
print sess.run(c_l2_loss)

for i in range(2):
    tower_name = '%s_%d' % ( 'tower' , i )
    print tower_name
    loss_name = re.sub('%s_[0-9]*/' % tower_name, '', c.op.name)
    print loss_name
    a=tf.get_default_graph().get_tensor_by_name('tmp_placeholder:0')
a=sess.run(tmp_holder , feed_dict={a:np.zeros([3])})
print a