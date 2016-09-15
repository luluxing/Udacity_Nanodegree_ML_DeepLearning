import pylearn2, os, struct, numpy
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import serial
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
from pylab import *
from numpy import *

# code used from http://g.sweyla.com/blog/2012/mnist-numpy/
def load_mnist(dataset="training", digits=numpy.arange(10), path="."):
    """
    Loads MNIST files into 2D numpy arrays

    Modified from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py

    Axis 0 of this array separate the different images, which means 

    we can get a 1D array of the kth image by images[k-1] 
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'mnist', 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'mnist', 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 'mnist', 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'mnist', 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows*cols), dtype=uint8) # create 1-D array
    labels = zeros((N, 10), dtype=int8) # Y array is stored as a 2D, one-hot array 
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]) # create 1-D array
        labels[i][lbl[ind[i]]-1] = 1
    return images, labels


if __name__ == "__main__":
    train_images, train_labels = load_mnist('training')
    # less data for CV and more for training
    # cv_images.shape=(10000, 784), train_images.shape=(50000, 784)
    cv_images, cv_labels = train_images[50000:], train_labels[50000:]
    train_images, train_labels = train_images[:50000], train_labels[:50000]
    test_images, test_labels = load_mnist('testing')

    train = dense_design_matrix.DenseDesignMatrix(X=train_images, y=train_labels)
    test = dense_design_matrix.DenseDesignMatrix(X=test_images, y=test_labels)
    cv = dense_design_matrix.DenseDesignMatrix(X=cv_images, y=cv_labels)
    
    path = '.'
    serial.save(os.path.join(path, 'pylearn2_mnist_train.pkl'), train)
    serial.save(os.path.join(path, 'pylearn2_mnist_cv.pkl'), cv)
    serial.save(os.path.join(path, 'pylearn2_mnist_test.pkl'), test)