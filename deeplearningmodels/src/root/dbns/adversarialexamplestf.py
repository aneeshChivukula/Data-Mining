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


# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print(mnist)
os.chdir('/home/aneesh/Documents/Adversarial Learning Datasets/USPS Dataset')
TRAIN_SIZE = 7300
TEST_SIZE = 2100
BATCH_SIZE = TEST_SIZE
TRAIN_LOOPS = (TRAIN_SIZE / BATCH_SIZE) - 1

IMAGE_SIZE = 256
OUTPUT_SIZE = 10
LEARNING_RATE = 0.5

 
label_input = tf.placeholder(tf.float32, shape=[OUTPUT_SIZE])
feature_input = tf.placeholder(tf.float32, shape=[IMAGE_SIZE])

sess = tf.InteractiveSession()
    
# print('label_batch',StringIO.StringIO(label_batch).getvalue())
# print('feature_batch',feature_batch)
# with sess.as_default():
#     print(label_batch.eval())
#     print(sess.run(label_batch))
#     tf.Print(label_batch,[label_batch])
qtrain = tf.FIFOQueue(TRAIN_SIZE, [tf.float32, tf.float32], shapes=[[IMAGE_SIZE], [OUTPUT_SIZE]])
enqueue_train_op = qtrain.enqueue([feature_input, label_input])

qtest = tf.FIFOQueue(TEST_SIZE, [tf.float32, tf.float32], shapes=[[IMAGE_SIZE], [OUTPUT_SIZE]])
enqueue_test_op = qtest.enqueue([feature_input, label_input])

def load_and_enqueue(trainfilename, testfilename):
    # Multiple calls for multiple files
    with open(trainfilename,'r') as f:
        for line in f:
            sline = map(float,line.rstrip('\n').split(' ')[:-1])
            l = [0]*10
            l[int(sline[0])] = 1
            sess.run(enqueue_train_op, feed_dict={feature_input: sline[1:],
                                      label_input: l})
    with open(testfilename,'r') as f:
        for line in f:
            sline = map(float,line.rstrip('\n').split(' '))
            l = [0]*10
            l[int(sline[0])] = 1
            sess.run(enqueue_test_op, feed_dict={feature_input: sline[1:],
                                      label_input: l})


load_and_enqueue('zip.train','zip.test')

x = tf.Variable(tf.zeros([BATCH_SIZE, IMAGE_SIZE]))
W = tf.Variable(tf.zeros([IMAGE_SIZE, OUTPUT_SIZE]))
b = tf.Variable(tf.zeros([OUTPUT_SIZE]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.Variable(tf.zeros([BATCH_SIZE,OUTPUT_SIZE], dtype=tf.float32))

init_op = tf.initialize_all_variables()
sess.run(init_op)
# print(sess.run(x))
# print(sess.run(y_))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)


for i in range(TRAIN_LOOPS):
    feature_train_batch, label_train_batch = qtrain.dequeue_many(BATCH_SIZE)
    update_x_train = tf.assign(x, feature_train_batch)
    update_y___train = tf.assign(y_, label_train_batch)
    sess.run([update_x_train,update_y___train])
    sess.run(train_step)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

feature_test_batch, label__test_batch = qtrain.dequeue_many(BATCH_SIZE)
update_x_test = tf.assign(x, feature_test_batch)
update_y___test = tf.assign(y_, label__test_batch)
sess.run([update_x_test,update_y___test])
print(sess.run(accuracy))






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

'''