import os, scipy.io, numpy, csv, random
from pylab import *
from random import random, shuffle, seed

seed(7)

def load_svhn(dataset='train_32x32.mat', path="."):
	"""
	Load the '.mat' format data

	The [:,:,:,i] stands for the ith image
	"""
	if dataset == "train_32x32.mat":
		fname = os.path.join(path, 'svhn_cropped_num', 'train_32x32.mat')
	elif dataset == "test_32x32.mat":
		fname = os.path.join(path, 'svhn_cropped_num', 'test_32x32.mat')
	mat = scipy.io.loadmat(fname)
	label_y = mat['y']
	img_x = mat['X']
	num = img_x.shape[3]
	return img_x, label_y

def to_gray(img):
	"""
	Convert the RGB matrix to greyscale matrix

	The [i,:,:] stands for the ith grey image
	"""
	new_img = zeros((img.shape[3], 32, 32), dtype=uint8)
	N = img.shape[3]
	for i in range(N):
		# https://en.wikipedia.org/wiki/Grayscale
		new_img[i] = img[:,:,0,i]*0.2125+img[:,:,1,i]*0.7154+img[:,:,2,i]*0.0721		
	return new_img

def img_standardize(img):
	"""
	Subtract the mean and divide by the standard deviation
	"""
	N = img.shape[0]
	new_img = zeros((N, 32, 32), dtype=float)
	mean = np.mean(img, axis=(1,2), dtype=float)
	std = np.std(img, axis=(1,2), dtype=float)
	
	for i in range(N):
		new_img[i] = (img[i] - mean[i]) / std[i]
	return new_img

def flatter_image(img):
	N = img.shape[0]
	new_img = zeros((N, 32*32), dtype=float)
	for i in range(N):
		new_img[i] = img[i].reshape((1, 32*32))
	return new_img

def to_hot_vector(label):
	N = label.shape[0]
	new_labl = zeros((N, 10), dtype=int8)
	for i in range(N):
		new_labl[i][label[i][0]-1] = 1
	return new_labl


if __name__ == '__main__':
	train_x, train_y = load_svhn('train_32x32.mat')
	test_x, test_y = load_svhn('test_32x32.mat')
	train_x1, test_x1 = to_gray(train_x),  to_gray(test_x)
	train_x2, test_x2 = img_standardize(train_x1), img_standardize(test_x1)

	train_x3, test_x3 = flatter_image(train_x2), flatter_image(test_x2)
	train_y1, test_y1 = to_hot_vector(train_y), to_hot_vector(test_y)
	
	# train_x3.shape=(73257,1024), test_x3.shape=(26032,1024) 
	# validation set!!!!
	# trainset=63257, validation=10000
	combined = zip(train_x3, train_y1)
	shuffle(combined)
	valid_x, valid_y = train_x3[:10000], train_y1[:10000]
	train_x, train_y = train_x3[10000:], train_y1[10000:]
	w1 = open('trx','w')
	numpy.save(w1, train_x)
	w2 = open('try','w')
	numpy.save(w2, train_y)
	w3 = open('testx','w')
	numpy.save(w3, test_x3)
	w4 = open('testy','w')
	numpy.save(w4, test_y1)
	w5 = open('validx','w')
	numpy.save(w5, valid_x)
	w6 = open('validy','w')
	numpy.save(w6, valid_y)
	# print train_x.shape,train_x[:, :, :, 1].shape,train_x[:, :, 1, 1].shape
	# print to_hot_vector(train_y)[1]
	# imshow(train_x[:, :, :, 1])
	# show()
	
	# imshow(wh[1])
	# show()

