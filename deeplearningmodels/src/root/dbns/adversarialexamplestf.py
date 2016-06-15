import StringIO
import os
import pandas
import random
from sklearn import datasets, cross_validation, metrics
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics.classification import accuracy_score
import sys
from tensorflow.contrib import learn as skflow
import threading

import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from dataset import Parser


localparser = Parser()
InDir = "/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/SerializedObjectCategories/" 
train_files = tf.gfile.Glob(InDir+"train-*")
test_files = tf.gfile.Glob(InDir+"validation-*")

# print(train_files)
train_files_queue = tf.train.string_input_producer(train_files)
num_preprocess_threads = 4
num_readers = 2 # num_readers must be greater than or equal 2
examples_per_shard = 1024
batch_size = 32
input_queue_memory_factor = 16
height = 462
width = 581
depth = 3
numclasslabels = 2
keep_prob = 0.5
# On large data, use multiple preprocessing threads and buffer queue on train_files_queue, image_preprocessing

min_queue_examples = examples_per_shard * input_queue_memory_factor


train_examples_queue = tf.RandomShuffleQueue(
          capacity=min_queue_examples + 3 * batch_size,
          min_after_dequeue=min_queue_examples,
          dtypes=[tf.string])

test_examples_queue = tf.FIFOQueue(
          capacity=examples_per_shard + 3 * batch_size,
          dtypes=[tf.string])

enqueue_ops = []
for _ in range(num_readers):
    reader = tf.TFRecordReader()
    _, value = reader.read(train_files_queue)
    enqueue_ops.append(train_examples_queue.enqueue([value]))
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(train_examples_queue, enqueue_ops))
    example_serialized = train_examples_queue.dequeue()

train_images_and_labels = []

for thread_id in range(num_preprocess_threads):
    image_buffer, label_index = localparser.parse_example_proto(example_serialized)
    image = localparser.image_preprocessing(image_buffer)
    train_images_and_labels.append([image, label_index])

print(train_images_and_labels)
print(num_preprocess_threads)
print(batch_size)

images, label_index_batch = tf.train.batch_join(
        train_images_and_labels,
        batch_size=batch_size,
        capacity=2 * num_preprocess_threads * batch_size)
print(images)
print(label_index_batch)
# Get correct queueing across multiprocessing - enqueue_many=True needed?

 
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# x = tf.placeholder(tf.float32, [None, width,height,depth])
# y_ = tf.placeholder(tf.float32, [None, numclasslabels])
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))

x = tf.Variable(tf.zeros([batch_size,width,height,depth]))
y_ = tf.Variable(tf.zeros([batch_size,numclasslabels], dtype=tf.float32))

sess = tf.InteractiveSession()
# sess.run(tf.initialize_all_variables())
# y = tf.nn.softmax(tf.matmul(x, W) + b)
 
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 
# for i in range(1000):
#   batch_xs, batch_ys = mnist.train.next_batch(100)
#   sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#   
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
 
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
  
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
  
# x_image = tf.reshape(x, [-1,width,height,depth])
print(x)
print(W_conv1)

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
  
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
  
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([146*116*64, 1024])
b_fc1 = bias_variable([1024])
  
h_pool2_flat = tf.reshape(h_pool2, [-1, 146*116*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  
W_fc2 = weight_variable([1024, numclasslabels])
b_fc2 = bias_variable([numclasslabels])
  
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
  
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

# convolution and shaping tensors must match input tensor
# load x in batches by wrapping above dequeuing in functions
# print train and test accuracy
# train over multiple gpus
# Wrt queues and coordinators over optimizers assuming operations can be performed across loops via functions 
# pipeline should load all the images at runtime. followinf runtime configuration and checkpointing of inception_train.py
# must ensure deserialization and loading loops are working correctly in train_step.
# experiment with each code separately in case of doubt
# once serial training is correct check distributed training in inception_distributed_train.py
# once training is done, need to package data loading code to resue for testfiles 

update_x_train = tf.assign(x, images)
update_y___train = tf.assign(y_, label_index_batch)

sess.run([update_x_train,update_y___train])
sess.run(train_step)

# for i in range(20000):
#   batch = mnist.train.next_batch(50)
#   if i%100 == 0:
#     train_accuracy = accuracy.eval(feed_dict={
#         x:batch[0], y_: batch[1], keep_prob: 1.0})
#     print("step %d, training accuracy %g"%(i, train_accuracy))
#   train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#  
# print("test accuracy %g"%accuracy.eval(feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
 
# line = "7 -1 -1 -1 -1 -1 -0.815 0.673 0.875 0.264 -0.376 -0.962 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.136 0.465 -0.844 -0.282 0.466 0.611 -0.918 -1 -1 -1 -1 -1 -1 -1 -1 -0.919 0.895 0.666 -1 -1 -1 0.651 -0.407 -1 -1 -1 -1 -1 -1 -1 -1 -0.233 0.51 -1 -1 -1 -1 0.515 0.087 -1 -1 -1 -1 -1 -1 -1 -1 -0.398 -0.496 -1 -1 -1 -1 0.216 0.093 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.503 -0.324 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.951 0.968 -0.676 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.289 0.761 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.616 -0.047 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.548 0.96 -0.842 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.311 0.16 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.832 1 -0.664 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.072 0.302 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.805 0.874 -0.727 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.288 -0.076 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.992 0.293 -1 -1 -1 -1 -1 -1 -1 -1 -1"
# splitline = line.split(' ')
# image7 = np.array(splitline[1:],np.float64)
# image = image7
# 
# # line = "9 -1 -1 -1 -1 -1 -0.948 -0.561 0.148 0.384 0.904 0.29 -0.782 -1 -1 -1 -1 -1 -1 -1 -1 -0.748 0.588 1 1 0.991 0.915 1 0.931 -0.476 -1 -1 -1 -1 -1 -1 -0.787 0.794 1 0.727 -0.178 -0.693 -0.786 -0.624 0.834 0.756 -0.822 -1 -1 -1 -1 -0.922 0.81 1 0.01 -0.928 -1 -1 -1 -1 -0.39 1 0.271 -1 -1 -1 -1 0.012 1 0.248 -1 -1 -1 -1 -1 -0.402 0.326 1 0.801 -0.998 -1 -1 -0.981 0.645 1 -0.687 -1 -1 -1 -1 -0.792 0.976 1 1 0.413 -0.976 -1 -1 -0.993 0.834 0.897 -0.951 -1 -1 -1 -0.831 0.14 1 1 0.302 -0.889 -1 -1 -1 -1 0.356 0.794 -0.836 -1 -0.445 0.074 0.833 1 1 0.696 -0.881 -1 -1 -1 -1 -1 -0.368 0.955 1 1 1 1 0.905 1 1 -0.262 -1 -1 -1 -1 -1 -1 -1 -0.507 0.451 0.692 0.692 -0.007 -0.237 1 0.882 -0.795 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.155 1 0.436 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.991 0.703 1 -0.025 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.833 0.959 1 -0.629 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.6 0.998 0.841 -0.932 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.424 1 0.732 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.908 0.43 0.622 -0.973 -1 -1 -1 -1 -1"
# # splitline = line.split(' ')
# # image9 = np.array(splitline[1:],np.float64)
# # image = image9
# 
# # imagediff = image9 - image7
# # image[70:134] = image[70:134] + imagediff[70:134]
# 
# ncols=16
# image.shape = (image.size//ncols, ncols)
# img = Image.fromarray(image)
# draw = ImageDraw.Draw(img) 
# draw.line((4,4, 7,7))
# 
# image = np.array(img)
# # plt.imshow(image, cmap='Greys_r')
# # plt.show()





# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print(mnist)
# os.chdir('/home/aneesh/Documents/Adversarial Learning Datasets/USPS Dataset')
# TRAIN_SIZE = 7300
# TEST_SIZE = 2100
# BATCH_SIZE = TEST_SIZE
# TRAIN_LOOPS = (TRAIN_SIZE / BATCH_SIZE) - 1
#  
# IMAGE_SIZE = 256
# OUTPUT_SIZE = 10
# LEARNING_RATE = 0.5
#  
#   
# label_input = tf.placeholder(tf.float32, shape=[OUTPUT_SIZE])
# feature_input = tf.placeholder(tf.float32, shape=[IMAGE_SIZE])
#  
# sess = tf.InteractiveSession()
#      
# # print('label_batch',StringIO.StringIO(label_batch).getvalue())
# # print('feature_batch',feature_batch)
# # with sess.as_default():
# #     print(label_batch.eval())
# #     print(sess.run(label_batch))
# #     tf.Print(label_batch,[label_batch])
# qtrain = tf.FIFOQueue(TRAIN_SIZE, [tf.float32, tf.float32], shapes=[[IMAGE_SIZE], [OUTPUT_SIZE]])
# enqueue_train_op = qtrain.enqueue([feature_input, label_input])
#  
# qtest = tf.FIFOQueue(TEST_SIZE, [tf.float32, tf.float32], shapes=[[IMAGE_SIZE], [OUTPUT_SIZE]])
# enqueue_test_op = qtest.enqueue([feature_input, label_input])
#  
# def load_and_enqueue(trainfilename, testfilename):
#     # Multiple calls for multiple files
#     with open(trainfilename,'r') as f:
#         for line in f:
#             sline = map(float,line.rstrip('\n').split(' ')[:-1])
#             l = [0]*10
#             l[int(sline[0])] = 1
#             sess.run(enqueue_train_op, feed_dict={feature_input: sline[1:],
#                                       label_input: l})
#     with open(testfilename,'r') as f:
#         for line in f:
#             sline = map(float,line.rstrip('\n').split(' '))
#             l = [0]*10
#             l[int(sline[0])] = 1
#             sess.run(enqueue_test_op, feed_dict={feature_input: sline[1:],
#                                       label_input: l})
#  
#  
# load_and_enqueue('zip.train','zip.test')
#  
# x = tf.Variable(tf.zeros([BATCH_SIZE, IMAGE_SIZE]))
# W = tf.Variable(tf.zeros([IMAGE_SIZE, OUTPUT_SIZE]))
# b = tf.Variable(tf.zeros([OUTPUT_SIZE]))
# y = tf.nn.softmax(tf.matmul(x, W) + b)
# y_ = tf.Variable(tf.zeros([BATCH_SIZE,OUTPUT_SIZE], dtype=tf.float32))
#  
# init_op = tf.initialize_all_variables()
# sess.run(init_op)
# # print(sess.run(x))
# # print(sess.run(y_))
#  
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
#  
#  
# for i in range(TRAIN_LOOPS):
#     feature_train_batch, label_train_batch = qtrain.dequeue_many(BATCH_SIZE)
#     update_x_train = tf.assign(x, feature_train_batch)
#     update_y___train = tf.assign(y_, label_train_batch)
#     sess.run([update_x_train,update_y___train])
#     sess.run(train_step)
#  
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#  
# feature_test_batch, label__test_batch = qtrain.dequeue_many(BATCH_SIZE)
# update_x_test = tf.assign(x, feature_test_batch)
# update_y___test = tf.assign(y_, label__test_batch)
# sess.run([update_x_test,update_y___test])
# print(sess.run(accuracy))






'''

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


Tensorflow, yadlt, skflow, sklearn
construct and execute computations as dataflow graphs(extending pig pipelines)
only dnn(without rbms) than dbn(with rbms) available in tensorflow. alternative is dae, cnn, rnn
following tutorials available on dnn
https://github.com/tensorflow/skflow
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/skflow
https://medium.com/@ilblackdragon/tensorflow-tutorial-part-1-c559c63c0cb1#.dwcg40up3
https://medium.com/@ilblackdragon/tensorflow-tutorial-part-2-9ffe47049c92#.hm7ztdr0w
http://terrytangyuan.github.io/2016/03/14/scikit-flow-intro/
https://github.com/pkmital/tensorflow_tutorials
https://github.com/nlintz/TensorFlow-Tutorials
https://www.tensorflow.org/versions/r0.9/get_started/basic_usage.html
https://www.tensorflow.org/versions/r0.9/how_tos/using_gpu/index.html
https://www.tensorflow.org/versions/r0.9/api_docs/python/framework.html#Graph
https://www.tensorflow.org/versions/r0.9/api_docs/python/client.html#session-management
https://www.tensorflow.org/versions/r0.9/how_tos/using_gpu/index.html
https://www.tensorflow.org/versions/r0.9/tutorials/deep_cnn/index.html
https://www.tensorflow.org/versions/r0.9/how_tos/distributed/index.html
https://www.tensorflow.org/versions/r0.9/resources/dims_types.html
https://www.tensorflow.org/versions/r0.9/tutorials/mnist/tf/index.html
Deep belief network with tensorflow
https://www.reddit.com/r/MachineLearning/comments/3z4b38/deep_belief_network_with_tensorflow/
https://github.com/blackecho/Deep-Learning-TensorFlow
http://deep-learning-tensorflow.readthedocs.io/en/latest/
https://groups.google.com/a/tensorflow.org/forum/#!forum/discuss

Pending
Done
tensorflow api and tutorials
theano api and tutorials
working dbn code
nolearn both versions
pylearn2 rbms
plato, crino, blocks on theano, lasagne
DBN in DL4J and R
To Do
adversarial examples on image data and text data - put train and test data in tensor format
cnn and dbn deep learning code on tensorflow, dl4j and theano
https://www.tensorflow.org/versions/r0.7/tutorials/mnist/download/index.html


with tf.Session() as sess:
  with tf.device("/gpu:1"):
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])
    product = tf.matmul(matrix1, matrix2)
    ...

What is the difference between a neural network and a deep belief network?
http://stats.stackexchange.com/questions/51273/what-is-the-difference-between-a-neural-network-and-a-deep-belief-network

# t = threading.Thread(target=load_and_enqueue)
# t.start()

# if __name__ == '__main__':
#     print ""
#     load_and_enqueue()

# matrix1 = tf.constant([[3., 3.]])
# matrix2 = tf.constant([[2.],[2.]])
# product = tf.matmul(matrix1, matrix2)
# sess = tf.Session()
# 
# result = sess.run(product)
# print(result)
# sess.close()
# 
# with tf.Session() as sess:
#     result = sess.run([product])
#     print(result)
# 
# os.chdir('/home/aneesh/tf_examples-master')
# data = pandas.read_csv('data/titanic_train.csv')
# print(data.shape)
# print(data.columns)
# y, X = data['Survived'], data[['Age', 'SibSp', 'Fare']].fillna(0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# lr = LogisticRegression()
# lr.fit(X_train, y_train)
# print accuracy_score(y_test, lr.predict(X_test))
# 
# random.seed(42)
# digits = datasets.load_digits()
# X = digits.images
# y = digits.target
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
#     test_size=0.2, random_state=42)
# 
# def conv_model(X, y):
#     X = tf.expand_dims(X, 3)
#     features = tf.reduce_max(skflow.ops.conv2d(X, 12, [3, 3]), [1, 2])
#     features = tf.reshape(features, [-1, 12])
#     return skflow.models.logistic_regression(features, y)
# 
# classifier = skflow.TensorFlowEstimator(model_fn=conv_model, n_classes=10,
#                                         steps=500, learning_rate=0.05,
#                                         batch_size=128)
# # classifier = skflow.TensorFlowLinearClassifier(n_classes=2, batch_size=128, steps=500, learning_rate=0.05)
# classifier.fit(X_train, y_train)
# score = metrics.accuracy_score(classifier.predict(X_test), y_test)
# print('Accuracy: %f' % score)




Draws images based on the MNIST data : Use matplotlib, pil and opencv in linux to plot and change images, draw line using pil or Pillow, estimate starting and ending pixel for line by looking at the image used for initializing the deep learning model, look at one picture to estimate start/end pixel and use same start/end point for all pictures, 
https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/fig/mnist.py

sudo pip install --upgrade Pillow
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image, ImageDraw

os.chdir('/home/aneesh/Documents/Adversarial Learning Datasets/USPS Dataset')
images = open('train.0.txt').readlines()

line = "7 -1 -1 -1 -1 -1 -0.815 0.673 0.875 0.264 -0.376 -0.962 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.136 0.465 -0.844 -0.282 0.466 0.611 -0.918 -1 -1 -1 -1 -1 -1 -1 -1 -0.919 0.895 0.666 -1 -1 -1 0.651 -0.407 -1 -1 -1 -1 -1 -1 -1 -1 -0.233 0.51 -1 -1 -1 -1 0.515 0.087 -1 -1 -1 -1 -1 -1 -1 -1 -0.398 -0.496 -1 -1 -1 -1 0.216 0.093 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.503 -0.324 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.951 0.968 -0.676 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.289 0.761 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.616 -0.047 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.548 0.96 -0.842 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.311 0.16 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.832 1 -0.664 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.072 0.302 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.805 0.874 -0.727 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.288 -0.076 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.992 0.293 -1 -1 -1 -1 -1 -1 -1 -1 -1"
splitline = line.split(' ')
image7 = np.array(splitline[1:],np.float64)
line = "9 -1 -1 -1 -1 -1 -0.948 -0.561 0.148 0.384 0.904 0.29 -0.782 -1 -1 -1 -1 -1 -1 -1 -1 -0.748 0.588 1 1 0.991 0.915 1 0.931 -0.476 -1 -1 -1 -1 -1 -1 -0.787 0.794 1 0.727 -0.178 -0.693 -0.786 -0.624 0.834 0.756 -0.822 -1 -1 -1 -1 -0.922 0.81 1 0.01 -0.928 -1 -1 -1 -1 -0.39 1 0.271 -1 -1 -1 -1 0.012 1 0.248 -1 -1 -1 -1 -1 -0.402 0.326 1 0.801 -0.998 -1 -1 -0.981 0.645 1 -0.687 -1 -1 -1 -1 -0.792 0.976 1 1 0.413 -0.976 -1 -1 -0.993 0.834 0.897 -0.951 -1 -1 -1 -0.831 0.14 1 1 0.302 -0.889 -1 -1 -1 -1 0.356 0.794 -0.836 -1 -0.445 0.074 0.833 1 1 0.696 -0.881 -1 -1 -1 -1 -1 -0.368 0.955 1 1 1 1 0.905 1 1 -0.262 -1 -1 -1 -1 -1 -1 -1 -0.507 0.451 0.692 0.692 -0.007 -0.237 1 0.882 -0.795 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.155 1 0.436 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.991 0.703 1 -0.025 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.833 0.959 1 -0.629 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.6 0.998 0.841 -0.932 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.424 1 0.732 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -0.908 0.43 0.622 -0.973 -1 -1 -1 -1 -1"
splitline = line.split(' ')
image9 = np.array(splitline[1:],np.float64)

image = image7
image = image9
imagediff = image9 - image7
image[70:134] = image[70:134] + imagediff[70:134]

ncols=16
image.shape = (image.size//ncols, ncols)

img = Image.fromarray(image)
draw = ImageDraw.Draw(img) 
draw.line((4,4, 7,7))

image = np.array(img)
plt.imshow(image, cmap='Greys_r')
plt.show()

http://matplotlib.org/users/image_tutorial.html

'''