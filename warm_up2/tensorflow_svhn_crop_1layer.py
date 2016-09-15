import tensorflow as tf
import numpy as np
import sklearn, random
from sklearn import metrics
import matplotlib.pyplot as plt

# define the model
x = tf.placeholder(tf.float32, shape=[None, 1024])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
svhn_tr = np.load('trx')
svhn_la = np.load('try')
svhn_validx = np.load('validx')
svhn_validy = np.load('validy')

def next_batch(n, last):
	a = last % 63257
	b = (n+last) % 63257
	if a >= b:
		return 63206, 63256, 0
	return a, b, n+last

def next_batch_distortion(images, labels, n, last, sess):
	a = last % 63257
	b = (n+last) % 63257
	r = n+last
	if a >= b:
		a, b, r =  63206, 63256, 0
	# reshape the images to be [batchsize,imgsize,imgsize]
	new_img = np.zeros((n, 32, 32), dtype=float)
	for i in range(n):
		new_img[i] = images[a+i].reshape((1, 32, 32))
	new_img = tf.image.random_flip_left_right(new_img)
	new_img = tf.image.random_brightness(new_img, max_delta=63)
  	new_img = tf.image.random_contrast(new_img, lower=0.2, upper=1.8)
  	float_image = tf.image.per_image_whitening(new_img)
  	float_image = float_image.eval(session=sess)
  	new_img2 = np.zeros((n, 32*32), dtype=float)
  	for j in range(n):
  		new_img2[j] = float_image[j].reshape((1, 32*32))
  	return new_img2, labels[a:b], r


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(x, [-1,32,32,1])

# one convolutional layer
W_conv1 = weight_variable([5, 5, 1, 64])
b_conv1 = bias_variable([64])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# densely connected layer
W_fc1 = weight_variable([16 * 16 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool1_flat = tf.reshape(h_pool1, [-1, 16 * 16 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer / softmax layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# cost function (ADAM optimizer)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

learning_rate = 1e-4 
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.initialize_all_variables())

curr = 0 # last image visited
train_acc = []
valid_acc = []
iteration = 10000
# saver.restore(sess, "model_1layer.ckpt")
# print 'Model restored!'
prev = 0
for i in range(iteration):
	a, b, curr = next_batch(50, curr)
	train_step.run(session=sess, feed_dict={x: svhn_tr[a:b], y_: svhn_la[a:b], keep_prob: 0.9})
	train_accuracy = accuracy.eval(session=sess, feed_dict={x: svhn_tr[a:b], y_: svhn_la[a:b], keep_prob: 1.0})	
	train_acc.append(train_accuracy) # plot the training accuracy 
	valid_correct = 0
	if i%10 == 0:
		for j in range(200):
			valid_accuracy = accuracy.eval(session=sess, feed_dict={x: svhn_validx[50*j:50*(j+1)], y_: svhn_validy[50*j:50*(j+1)], keep_prob: 1.0})
			valid_correct += valid_accuracy * 50
		validacc = valid_correct/10000.
		valid_acc.append(validacc)
		print "step %d, training accuracy %g"%(i, train_accuracy)
		print "step %d, validation accuracy %g"%(i, validacc)
	if abs(prev-validacc) <= 1e-10:
		save_path = saver.save(sess, "model_1layer_2.ckpt")
		print("Model saved in file: %s" % save_path)
		break
	prev = validacc
	

svhn_testx = np.load('testx')
svhn_testy = np.load('testy')

y_true = []
for i in range(len(svhn_testy)):
	y_true += svhn_testy[i].nonzero()[0].tolist()
y_p = tf.argmax(y_conv, 1)

correct = 0
y_pred = []
i = 0
while i+50 < svhn_testx.shape[0]:
	test_accuracy, y_pred1 = sess.run([accuracy, y_p], feed_dict={x: svhn_testx[i:(i+50)], y_: svhn_testy[i:(i+50)], keep_prob: 1.0})
	y_pred += y_pred1.tolist()
	i = i+50

test_accuracy, y_pred1 = sess.run([accuracy, y_p], feed_dict={x: svhn_testx[i:svhn_testx.shape[0]], y_: svhn_testy[i:svhn_testx.shape[0]], keep_prob: 1.0})
y_pred += y_pred1.tolist()


print 'test accuracy', sum([y_true[i]==y_pred[i] for i in range(len(y_true))])*1./len(y_true)
print 'precision', sklearn.metrics.precision_score(y_true, y_pred)
print "Recall", sklearn.metrics.recall_score(y_true, y_pred)
print "f1_score", sklearn.metrics.f1_score(y_true, y_pred)
print "confusion_matrix"
w = open('confusion_1layer','w')
b=np.save(w, sklearn.metrics.confusion_matrix(y_true, y_pred))

x_axes = [j for j in range(len(train_acc))]
plt.plot(x_axes, train_acc, marker='o', linestyle='--', color='r', label='Train Accuracy')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.title('Train Accuracy')
plt.legend()
plt.savefig('train_1layer.png')
# plt.show()