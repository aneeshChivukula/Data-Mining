# from PIL import Image, ImageDraw
# import StringIO
# from sklearn import datasets, cross_validation, metrics
# import sklearn
# from sklearn.cross_validation import train_test_split
# from sklearn.linear_model.logistic import LogisticRegression
# from sklearn.metrics.classification import accuracy_score
# from tensorflow.contrib import learn as skflow
# import pandas

import os
import math
import random
import sys
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
from root.cnns.decodedataset import Parser
import tensorflow as tf


localparser = Parser()
# InDir = "/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/SerializedObjectCategories/" 
# test_dir = '/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/101_ObjectCategories_Test'
# labels_file = '/home/aneesh/models-master/inception/labels.txt'
# train_batch_size = 100

InDir = "/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/SerializedClasses/" 
test_dir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TestSplit/'
labels_file = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/Labels.txt'
train_batch_size = 100

train_files = tf.gfile.Glob(InDir+"train-*")
test_files = tf.gfile.Glob(InDir+"test-*")
# validation_files = tf.gfile.Glob(InDir+"validation-*")

# test_batch_size = 50
# batch_size = 3000
num_preprocess_threads = 4
training_iters = 2
# training_iters = 50
# training_iters = 100
display_step = 10
learning_rate= 0.0001

# n_input = 90000
n_classes = 2
# keep_prob = 1.
training_keep_prob = 0.5
testing_keep_prob = 1.

# stride = 4
denselayernumneurons = 4096
# denselayernumneurons2 = 1000
# convmapsize = 11
height = 224
width = 224
depth = 3
# height = 100
# width = 300
# denselayernumneurons = 200
# convmapsize = 5
# stride = 1

layer1filters = 64
layer2filters = 192
layer3filters = 384
layer4filters = 256
layer5filters = 256

layer1convmapsize = 11
layer2convmapsize = 5
layer3convmapsize = 3
layer4convmapsize = 3
layer5convmapsize = 3

convimagesize = 3

class ImageCoder(object):
    def __init__(self):
        self._sess = tf.Session()
    
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
        
    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
        return image

def weight_variable(shape,stddev):
  initial = tf.truncated_normal(shape, stddev)
  return tf.Variable(initial)
  
def bias_variable(shape,const):
  initial = tf.constant(const, shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W,stride,currpadding):
   return tf.nn.conv2d(x, W, padding=currpadding, strides=[1, stride, stride, 1])
   
def max_pool_2d(x,size,stride,currpadding):
  return tf.nn.max_pool(x, padding=currpadding, ksize=[1, size, size, 1],
                        strides=[1, stride, stride, 1])  

def read_and_decode(filenames_queue):
    reader = tf.TFRecordReader()
    _, value = reader.read(filenames_queue)
    image_buffer, label_index = localparser.parse_example_proto(value)
    image = localparser.image_preprocessing(image_buffer)
#     return image, tf.reshape(label_index, shape = [1,1])
    return image, label_index


def inputs(filenames):
    train_filenames_queue = tf.train.string_input_producer(filenames)
    train_image, train_onehot_label = read_and_decode(train_filenames_queue)
    train_images, train_labels = tf.train.batch([train_image, train_onehot_label], train_batch_size, num_threads=num_preprocess_threads, capacity=2*train_batch_size)
    print('Loaded data')
    return train_images, train_labels
# Prefer to Have only one tf file for training. And one tf file for testing
# Change batch size to train data size

def model(x,y_,dropout,weights_variables_dict):
    
    print(x.dtype.base_dtype)
    print(weights_variables_dict["W_conv1"].dtype.base_dtype)

    
    h_conv1 = tf.nn.relu(conv2d(x, weights_variables_dict["W_conv1"],4,'SAME') + weights_variables_dict["b_conv1"])
    h_conv1_normed = tf.nn.local_response_normalization(h_conv1,5,2,0.0001,0.75)
    h_pool1 = max_pool_2d(h_conv1_normed,3,2,'SAME')
     
    h_conv2 = tf.nn.relu(conv2d(h_pool1, weights_variables_dict["W_conv2"],1,'VALID') + weights_variables_dict["b_conv2"])
    h_conv2_normed = tf.nn.local_response_normalization(h_conv2,5,2,0.0001,0.75)
    h_pool2 = max_pool_2d(h_conv2_normed,3,2,'SAME')
     
    h_conv3 = tf.nn.relu(conv2d(h_pool2, weights_variables_dict["W_conv3"],1,'VALID') + weights_variables_dict["b_conv3"])
    h_conv4 = tf.nn.relu(conv2d(h_conv3, weights_variables_dict["W_conv4"],1,'VALID') + weights_variables_dict["b_conv4"])

    h_conv5 = tf.nn.relu(conv2d(h_conv4, weights_variables_dict["W_conv5"],1,'VALID') + weights_variables_dict["b_conv5"])
    h_pool3 = max_pool_2d(h_conv5,3,2,'SAME')
    
#     print(h_pool3)
#     sys.exit()
     
    h_pool3_flat = tf.reshape(h_pool3, [-1, convimagesize*convimagesize*layer5filters]) # Must change 4*4*layer3filters to correct shape for tf.matmul
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, weights_variables_dict["W_fc1"]) + weights_variables_dict["b_fc1"])
    h_fc1_drop = tf.nn.dropout(h_fc1, dropout)
    
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, weights_variables_dict["W_fc2"]) + weights_variables_dict["b_fc2"])
    h_fc2_drop = tf.nn.dropout(h_fc2, dropout)
     
    y_conv=tf.nn.softmax(tf.matmul(h_fc2_drop, weights_variables_dict["W_fc3"]) + weights_variables_dict["b_fc3"])
     
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv,1e-10,1.0)), reduction_indices=[1]))  
#     cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))  
     
    return (cross_entropy,y_conv)


def trainingAndtesting(train_images,train_labels,test_images,test_labels):
# def trainingAndtesting(x,y_):

    print('trainingAndtesting model - start')
    weights_variables_dict = {
        "W_conv1" : weight_variable([layer1convmapsize, layer1convmapsize, depth, layer1filters],0.01),                     
        "W_conv2" : weight_variable([layer2convmapsize, layer2convmapsize, layer1filters, layer2filters],0.01),
        "W_conv3" : weight_variable([layer3convmapsize, layer3convmapsize, layer2filters, layer3filters],0.03),
        "W_conv4" : weight_variable([layer4convmapsize, layer4convmapsize, layer3filters, layer4filters],0.03),
        "W_conv5" : weight_variable([layer5convmapsize, layer5convmapsize, layer4filters, layer5filters],0.03),

        "b_conv1" : bias_variable([layer1filters],1.),
        "b_conv2" : bias_variable([layer2filters],1.),
        "b_conv3" : bias_variable([layer3filters],1.),
        "b_conv4" : bias_variable([layer4filters],1.),
        "b_conv5" : bias_variable([layer5filters],1.),
        
        
        # Continue from here
        "W_fc1" : weight_variable([convimagesize*convimagesize*layer5filters, denselayernumneurons],0.01),
        "W_fc2" : weight_variable([denselayernumneurons, denselayernumneurons],0.01),
        "W_fc3" : weight_variable([denselayernumneurons, n_classes],0.01),

        "b_fc1" : bias_variable([denselayernumneurons],1.),
        "b_fc2" : bias_variable([denselayernumneurons],1.),        
        "b_fc3" : bias_variable([n_classes],-7.)
    #     "keep_prob" : tf.Variable(0,dtype=tf.float32)                      
    #     "test_batch_x": tf.Variable(tf.zeros([train_batch_size,height,width,depth])),
    #     "test_batch_y_": tf.Variable(tf.zeros([test_batch_size,height,width,depth]))
    }
    keep_prob = tf.placeholder(tf.float32) 
    x = tf.placeholder(tf.float32, [None, height,width,depth])
    y_ = tf.placeholder(tf.float32, [None, n_classes])
    

#     keep_prob = tf.Variable(tf.constant(0.),dtype=tf.float32)
#     keep_prob = training_keep_prob
    cost,y_conv = model(x,y_,keep_prob,weights_variables_dict)
#     cost,y_conv = model(x,y_)
 
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    # Setting only learning_rate in Optimizer. Remaining parameters are paper specific optimizations.
 
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Need to check the network architecture since accuracy is varying from 0.1 to 0.8 across runs
    # Check dimensions of y_ and y as expected
     
     
    sess = tf.Session()
    coord = tf.train.Coordinator()
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    print('Initialized data')
    # Visualize graph ops created uptil here
     
    try:
        step = 0
        while not coord.should_stop(): 
            start_time = time.time()
             
            sess.run(train_step,feed_dict={x:sess.run(train_images),y_:sess.run(train_labels),keep_prob: training_keep_prob})
            print('Training model') 
 
            duration = time.time() - start_time
             
            if step % display_step == 0:
#                 keep_prob = 1
                loss,acc = sess.run([cost,accuracy],feed_dict={x:sess.run(train_images),y_:sess.run(train_labels),keep_prob: training_keep_prob})
#                 loss,acc = sess.run([cost,accuracy], feed_dict={x:sess.run(x),y_:sess.run(y_)})
#                 loss,acc = sess.run([cost,accuracy], feed_dict={x: x,y_: y_})
#                 print('sess.run(x)',sess.run(x))
#                 print('sess.run(y_)',sess.run(y_))
                
                print('Step %d: minibatch loss = %.6f training accuracy = %.2f (%.3f sec)' % (step,loss,acc,duration))
            if(step>training_iters): # Train the same network over multiple runs
                coord.request_stop()
            step += 1
    # Train for multiple iterations before computing accuracy
    # We can also train on multiple batches because computation graph parameters are updated across batches
    except tf.errors.OutOfRangeError:
        print('Exiting after training for %d steps.',step)           
    finally:
         
#         keep_prob = 1.
#         keep_prob = testing_keep_prob
        print('Step %d: testing accuracy = %.2f' % (step,sess.run(accuracy, feed_dict={x:sess.run(test_images),y_:sess.run(test_labels),keep_prob: testing_keep_prob})))
         
         
        coord.request_stop()
        coord.join(threads)
#         coord.request_stop()
#         coord.join(threads)

    sess.close()
    print('trainingAndtesting model - end')
    
def run_training():
    with tf.Graph().as_default():
        curr_dir = test_dir

        
        unique_labels = [l.strip() for l in tf.gfile.FastGFile(labels_file, 'r').readlines()]
        filenames = []
        labels = []
        humans = []

        label_index = 0

        for synset in unique_labels:
            jpeg_file_path = '%s/%s/*' % (curr_dir,synset)
            matching_files = tf.gfile.Glob(jpeg_file_path)
            filenames.extend(matching_files)
            labels.extend([label_index] * len(matching_files))
            humans.extend([synset] * len(matching_files))
            label_index += 1
        
        test_images = []
        test_labels = []
        for i in xrange(0,len(filenames)-10):
            filename = filenames[i]
            label = labels[i]
            human = humans[i]
            
            image_data = tf.gfile.FastGFile(filename, 'r').read()
            image = tf.image.decode_jpeg(image_data, channels=depth)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.reshape(image, shape=[height, width, depth])
            test_images.append(image)  
            test_labels.append(tf.cast(tf.reshape(tf.one_hot(label,depth = 2) , shape=[2]), dtype=tf.int32))  
        test_labels = tf.cast(test_labels, dtype=tf.float32)

        
        train_images, train_labels = inputs(train_files)
        train_labels = tf.cast(train_labels, dtype=tf.float32)        
        
        
        trainingAndtesting(train_images,train_labels,test_images,test_labels)

def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()


