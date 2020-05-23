import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


# Parameters
a = tf.Variable([.3], tf.float32)
b  = tf.Variable([-.3], tf.float32)


# Input and output
x = tf.placeholder(tf.float32)

linear_module = a * x + b

y = tf.placeholder(tf.float32)

squared_delta = tf.square()

sess = tf.Session()

print(sess.run(adder_node, {a: [1,3], b:[2,4]}))