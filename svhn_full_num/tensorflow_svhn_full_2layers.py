import tensorflow as tf
import numpy as np
import sklearn, os
from sklearn import metrics
from random import randint
from sklearn.cross_validation import KFold

# import the dataset
# svhn_tr = np.load('outtrx')
# svhn_la = np.load('outtry')
# svhn_validx = np.load('validx')
# svhn_validy = np.load('validy')

# 10-fold cross validation
svhn_data = np.load('outtrx')
svhn_test = np.load('outtry')
svhn_testx = np.load('outtestx')
svhn_testy = np.load('outtesty')


# define the model
x = tf.placeholder(tf.float32, shape=[None, 1024])
y_ = tf.placeholder(tf.int64, shape=[None, 5])

def next_batch(n, last):
	# a = last % 25402
	# b = (n+last) % 25402
	a = last % 9000
	b = (n+last) % 9000
	if a >= b: # return the last batch size images
		# return 25352, 25402, 0
		return 8950, 9000, 0
	return a, b, n+last

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

# first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 64])
b_conv1 = bias_variable([64])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([5, 5, 64, 128])
b_conv2 = bias_variable([128])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
W_fc1 = weight_variable([8 * 8 * 128, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# build five classifiers on top of the network
clf_w, clf_b = [], []
for i in range(5):
	clf_w.append(weight_variable([1024, 11]))
	clf_b.append(bias_variable([11]))

# readout layer / softmax layer
logit_list = []
for i in range(5):
	logit_list.append(tf.matmul(h_fc1_drop, clf_w[i]) + clf_b[i])

# cost function (ADAM optimizer)
def cost_function(truelabl, prelabl):
	cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(prelabl, truelabl))
	return cross_entropy

cross_entropy = 0
for i in range(5):
	cross_entropy += cost_function(y_[:, i], logit_list[i])
  
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

y_predict = []
for i in range(5):
	y_predict += [tf.nn.softmax(logit_list[i])]	

y_p = tf.transpose(tf.argmax(y_predict, 2))
correct_prediction = tf.equal(y_, y_p)

def accuracy(pred, true):
	cols = true.shape[1]
	rows = true.shape[0]
	correct = 0
	for r in range(rows):
		tmp = pred[r] == true[r]
		if sum(tmp) == cols:
			correct += 1
	return 1.*correct/rows

# transform labels to five-number digit, e.g. [1,10,3,4,9] -> 10349
def to_num(l):
	"""
	Transfrom the list into the number
	"""
	result = []
	for i in l:
		tmp = 0
		for j in i:
			if j == 0:
				break
			tmp = tmp * 10 + j % 10
		result += [tmp]
	return result


steps = 100001
valid_accuracy = []

# define k-fold cross-validation
kf = KFold(36630, n_folds=10)
xx = 0
for train_index, test_index in kf:
	saver = tf.train.Saver()
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())

	svhn_tr, svhn_validx = svhn_data[train_index], svhn_data[test_index]
	svhn_la, svhn_validy = svhn_test[train_index], svhn_test[test_index]
	svhn_validy = to_num(svhn_validy)
	temp = [0]
	curr = 0 # last image visited
	flag = False

	save_name = "10fold_" + str(xx) + "_model_2layers_10fold.ckpt"
	save_path = os.path.join('.', 'kfold_checkpoint_2layers', save_name)

	for i in range(steps):
		a, b, curr = next_batch(50, curr)
		train_step.run(session=sess, feed_dict={x: svhn_tr[a:b], y_: svhn_la[a:b], keep_prob: 0.9})
		pred = sess.run(y_p, feed_dict={x: svhn_tr[a:b], keep_prob: 1.0})
		train_accuracy = accuracy(pred, svhn_la[a:b])
		# print "step %d, training accuracy %g"%(i, train_accuracy)
		if i%10 == 0:
			valid_pred = []
			for j in range(20):
				tmp = sess.run(y_p, feed_dict={x: svhn_validx[50*j:50*(j+1)], keep_prob: 1.0})
				valid_pred += tmp.tolist()
			valid_pred = to_num(valid_pred)
			validacc = sum([svhn_validy[k]==valid_pred[k] for k in range(len(svhn_validy))])*1./len(valid_pred)
			temp.append(validacc)
			print "step %d, training accuracy %g"%(i, train_accuracy)
			print "step %d, validation accuracy %g"%(i, validacc)			
		if abs(temp[-1]-temp[-2]) <= 1e-10:
			print xx, "fold valid accuracy stops diminishing!"
			save_file = saver.save(sess, save_path)
			print("Model saved in file: %s" % save_file)
			flag = True
			break

	if not flag:	
		print xx, "fold runs out of steps!"
		save_file = saver.save(sess, save_path)
		print("Model saved in file: %s" % save_file)
		
	# record the validation accuracy for further analysis		
	valid_accuracy.append(temp)
	print "Done with No.", xx, "10-fold cross validation!"
	xx += 1

valid_name = "10fold_1"
valid_path = os.path.join('.', 'kfold_accuracy_2layers', valid_name)
save_valid = open(valid_path, 'w')
np.save(save_valid, np.array(valid_accuracy))
	

		
y_true = []
for i in range(len(svhn_testy)):
	y_true.append(svhn_testy[i])

y_p = tf.transpose(tf.argmax(y_predict, 2))

y_pred = []
i = 0
while i+50 < svhn_testx.shape[0]:
	y_pred1 = sess.run(y_p, feed_dict={x: svhn_testx[i:(i+50)], keep_prob: 1.0})
	y_pred += y_pred1.tolist()
	i = i+50

y_pred1 = sess.run(y_p, feed_dict={x: svhn_testx[i:svhn_testx.shape[0]], keep_prob: 1.0})
y_pred += y_pred1.tolist()



y_true = to_num(y_true)
y_pred = to_num(y_pred)

print 'accuracy', sum([y_true[i]==y_pred[i] for i in range(len(y_true))])*1./len(y_true)
print 'precision', sklearn.metrics.precision_score(y_true, y_pred)
print "Recall", sklearn.metrics.recall_score(y_true, y_pred)
print "f1_score", sklearn.metrics.f1_score(y_true, y_pred)
m = sklearn.metrics.confusion_matrix(y_true, y_pred)
print "confusion_matrix"
w = open('confusion_2layers_10fold','w')
np.save(w, m)

