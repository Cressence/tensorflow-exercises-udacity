import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

node1 = tf.constant(5.0)
node2 = tf.constant(6.0)

c = node2 * node1

sess= tf.Session()

File_Writer = tf.summary.FileWriter('/Users/yakamcressence/Downloads/Work/Projects/other/python-mastercalss/graph', sess.graph)

print(sess.run(c))

sess.close()
