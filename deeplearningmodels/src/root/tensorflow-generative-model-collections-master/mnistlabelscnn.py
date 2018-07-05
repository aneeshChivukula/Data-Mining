'''
Created on 12 Feb. 2018

@author: 99176493


Deep MNIST Tutorials:
https://www.tensorflow.org/versions/r0.12/tutorials/mnist/pros/
https://gist.github.com/saitodev/c4c7a8c83f5aa4a00e93084dd3f848c5

https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python
https://nextjournal.com/a/17592186058848

IWGAN parameters:
nh=8
nw=16
N=128

# https://github.com/igul222/improved_wgan_training/blob/master/tflib/save_images.py
# https://docs.scipy.org/doc/scipy/reference/misc.html
# Load all train and test images in Iwgan : Each x has shape (28,28)

BEGAN, InfoGAN parameters:
nh=8
nw=8
N=64

# Load all test images only in Began and Infogan
# 8*8 image consisting each image 28, 28, 1
# Use same loader method with different input image shapes

# Each image as same flatten order as tensorflow mnist - assuming row-major order which is default for numpy flatten() - check other flatten orders of output labels are incorrect on inspection








cifar10_game:

load_gansamples

binarizermulti
binarizerganmulti
binarizer

trainmultiplayercnn==True
testgenadv==True
multiplayer==True

cifar_train:
max_steps



two label training and testing data available on disk, no need of load_gansamples,
binarizerganmulti in numpy makes bin files for cifar10
binarizermulti creates manipulated data for multiplayer game
binarizer creates manipulated data for twoplayer game

trainmultiplayercnn and multiplayer is not required for training and testing respectively on generated data
no need to use mnist data in the testing process for testgenadv, simply use the train and test split directories created for earlier experiments,

trace control flow in training and testing methods, corresponding flags and parameters in testgenadv code block,
check all parameters set appropriately in cifar_train and cifar_eval, any other methods called in either twoplayer or multiplayer tests,

run this command in workspace, run all experiments in deeplearningmodels_25 and deeplearningmodels_26, testgenadv code block is missing on server,
stdbuf -oL python cifar10_game.py > out-phoenix11-5and8.txt
tail -F out-phoenix11-5and8.txt

alphas are in /data/achivuku/Documents
code in /data/achivuku/workspace
copy data directories like cifar10_22 in /data/achivuku/Desktop

Store data in /scratch
Run code in /data


Updates:
ganimages, ganlabels to be of expected shape
copy remaining part of testgenadv from desktop to server
binarizerganmulti, cifar10_train.train(), binarizermulti, cifar10_eval.evaluate() to work as expected

images shape to be correctly maintained across the methods tested
change IMAGE_SIZE from 28 to 32 to be compatible with cifar10-input.py

'''
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import os, sys
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import cPickle as pickle

def loadgenerateddata(CurrDir,nh,nw,N):
    h, w = 28, 28
    allsamples = []

    for savedfile in sorted(os.listdir(CurrDir)):
        save_path = CurrDir + savedfile
        img = imread(save_path)

        X = []
        for n in xrange(0, N):
            i = n % nw
            j = n / nw
            x = img[j * h:j * h + h, i * w:i * w + w]
            X.append(x.ravel())
        allsamples.extend(X)

    return np.array(allsamples)

gendatadir = "/home/achivuku/Desktop/tensorflow-generative-model-collections-master/"

# CurrDir = "/home/achivuku/Desktop/improved_wgan_training-master/output/"
# genimagesfile = "iwgan_genimages.pkl"
# genlabelsfile = "iwgan_genlabels.pkl"
# nh, nw, N = 8, 16, 128
# iwgan_testimages = loadgenerateddata(CurrDir, nh, nw, N)

# CurrDir = "/home/achivuku/Desktop/tensorflow-generative-model-collections-master/results/BEGAN_mnist_64_62/output/"
# genimagesfile = "began_genimages.pkl"
# genlabelsfile = "began_genlabels.pkl"
# nh, nw, N = 8, 8, 64
# began_testimages = loadgenerateddata(CurrDir, nh, nw, N)

# print(began_testimages[0].shape)
# plt.imshow(began_testimages[9991].reshape(28,28))
# plt.show()

CurrDir = "/home/achivuku/Desktop/tensorflow-generative-model-collections-master/results/infoGAN_mnist_64_62/output/"
genimagesfile = "infogan_genimages.pkl"
genlabelsfile = "infogan_genlabels.pkl"
nh, nw, N = 8, 8, 64
infogan_testimages = loadgenerateddata(CurrDir, nh, nw, N)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
numgenimages = mnist.train.images.shape[0]

# labels = pickle.load(open(currlabelsfilepath,'rb'))
# print(mnist.train.images.shape[0])
# print(iwgan_testimages.shape)

# print(iwgan_testimages[0:10000])
# print(iwgan_testimages[0])
# print('label',labels[9991])
# plt.imshow(iwgan_testimages[9991].reshape(28,28))
# plt.show()





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

# Input layer
x  = tf.placeholder(tf.float32, [None, 784], name='x')
y_ = tf.placeholder(tf.float32, [None, 10],  name='y_')
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolutional layer 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Convolutional layer 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer 1
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob  = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected layer 2 (Output layer)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')

# Evaluation functions
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# Training algorithm
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Training steps
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  # max_steps = 10
  max_steps = 10000
  for step in range(max_steps):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    if (step % 100) == 0:
      print(step, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
  # print(max_steps, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
  # print(max_steps, sess.run(y, feed_dict={x: mnist.test.images, keep_prob: 1.0}))
  # labels = sess.run(tf.argmax(y, 1), feed_dict={x: mnist.test.images, keep_prob: 1.0})

  labels = []
  images = []
  for step in range(int(numgenimages / 10000)):
      # currimages = iwgan_testimages[step*10000:(step+1)*10000]
      # currimages = began_testimages[step*10000:(step+1)*10000]
      currimages = infogan_testimages[step*10000:(step+1)*10000]

      currlabels = sess.run(tf.argmax(y, 1), feed_dict={x: currimages, keep_prob: 1.0})
      images.extend(currimages)
      labels.extend(currlabels)
  # print('CNN predicted',len(labels))

  print('currimages.shape',currimages.shape)
  # print('images.shape',images[0:10])
  print('images.shape',len(images))


  pickle.dump(images, open(gendatadir + genimagesfile, 'wb'))
  pickle.dump(labels, open(gendatadir + genlabelsfile, 'wb'))






