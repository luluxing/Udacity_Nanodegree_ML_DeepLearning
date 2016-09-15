import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist/", one_hot=True)

# define the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W)+b)

# train the model
y_ = tf.placeholder(tf.float32, [None, 10])
# cost function (gradient descent optimizer)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# initialize the variables
init = tf.initialize_all_variables()

# lauch the model in a session and run the operation that initializes the variables
sess = tf.Session()
sess.run(init)

for i in range(1000): # run the training 1000 times!
	batch_xs, batch_ys = mnist.train.next_batch(100) # get a batch of 100 random data points
	sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

# evaluate the model
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
