import os, numpy, h5py, scipy.misc
from pylab import *
from random import random, shuffle, seed

seed(7)

def crop(img, t, b, l, r):
    lx, ly = img.shape
    b = min(b, lx)
    r = min(r, ly)
    return img[t:b, l:r]

def read_struct(dire):
    """
    Read the struct info of every image and

    Load the png file and crop it based on the struct file 

    Return the cropped-digits images and the labels   
    """
    # dire is either 'train' or 'test'
    fn = os.path.join('.', dire, 'digitStruct.mat')
    f = h5py.File(fn, 'r')
    variables = f.items()
    N = len(f['digitStruct']['name']) # number of the images

    discard_img = [] # store the discarded images' names
    data = numpy.zeros((N, 32, 32), dtype=float)
    labels = numpy.zeros((N, 5), dtype=int) # to distinguish from 1-10
    for j in range(N):
        curr = {}
        names = f[f['digitStruct']['name'][j][0]].value
        name = ''.join([chr(ch[0]) for ch in names])

        bbox = f['digitStruct']['bbox'][j].item()
        digit_num = len(f[bbox]['height'])

        # if the digits in the image are more than 5, discard this image
        if digit_num > 5:
            discard_img.append(name)
            continue
        curr['height'] = []
        curr['left'] = []
        curr['top'] = []
        curr['width'] = []
        if digit_num == 1:
            curr['height'].append(f[bbox]['height'].value[0][0])
            labels[j][0] = f[bbox]['label'].value[0][0]
            curr['left'].append(f[bbox]['left'].value[0][0])
            curr['top'].append(f[bbox]['top'].value[0][0])
            curr['width'].append(f[bbox]['width'].value[0][0])
        else:
            for i in range(digit_num):
                curr['height'].append(f[f[bbox]['height'].value[i].item()].value[0][0])
                labels[j][i] = f[f[bbox]['label'].value[i].item()].value[0][0]
                curr['left'].append(f[f[bbox]['left'].value[i].item()].value[0][0])
                curr['top'].append(f[f[bbox]['top'].value[i].item()].value[0][0])
                curr['width'].append(f[f[bbox]['width'].value[i].item()].value[0][0])
        
        imagepath = os.path.join('.', dire, name)
        imgfile = scipy.misc.imread(imagepath, flatten=True)
        rownum, colnum = imgfile.shape

        top_most = min(curr['top'])
        left_most = min(curr['left'])
        height_big = max(curr['top']) + curr['height'][curr['top'].index(max(curr['top']))] - top_most
        width_big = max(curr['left']) + curr['width'][curr['left'].index(max(curr['left']))] - left_most

        # expand this bounding box by 30% in both the x and the y direction
        # 15% in either positive or negative direction
        top_most -= 0.15 * height_big
        left_most -= 0.15 * width_big
        if top_most < 0: top_most = 0
        if left_most < 0: left_most = 0
        bottom_most = top_most + 1.15 * height_big
        right_most = left_most + 1.15 * width_big
        crop_img = crop(imgfile, top_most, bottom_most, left_most, right_most)
        # print j, top_most, bottom_most, left_most, right_most,'^&^)&*(_)(#*($)'
        # resize the cropped to 32-by-32 pixels
        resized_img = scipy.misc.imresize(crop_img, (32, 32))
        
        # standardize the image
        mean = numpy.mean(resized_img)
        std = numpy.std(resized_img)
        data[j] = (resized_img - mean) / std

    f.close()
    return data, labels

def flatten_image(img):
    N = img.shape[0]
    new_img = zeros((N, 32*32), dtype=float)
    for i in range(N):
        new_img[i] = img[i].reshape((1, 32*32))
    return new_img

testdata, testlabl = read_struct('test')
testdata = flatten_image(testdata)
w1 = open('outtestx', 'w')
w2 = open('outtesty', 'w')
numpy.save(w1, testdata)
numpy.save(w2, testlabl)


traindata, trainlabl = read_struct('train')
traindata = flatten_image(traindata)
combined = zip(traindata, trainlabl)
shuffle(combined)

# validdata, validlabl = traindata[:8000], trainlabl[:8000]
traindata, trainlabl = traindata[36630:], trainlabl[36630:]
w3 = open('outtrx', 'w')
w4 = open('outtry', 'w')
numpy.save(w3, traindata)
numpy.save(w4, trainlabl)
# w5 = open('validx', 'w')
# w6 = open('validy', 'w')
# numpy.save(w5, validdata)
# numpy.save(w6, validlabl)




