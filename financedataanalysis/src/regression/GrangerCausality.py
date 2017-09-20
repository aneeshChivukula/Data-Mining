'''
Created on 10 Aug 2017

@author: aneesh
'''
# import scipy.io as sio
import os

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys

sys.path.append('..')

import tensorflow as tf

# DATA_PATH = '/home/aneesh/Desktop/UTS Literature Survey/Deep Learning Software/Sourcecodes/stanford-tensorflow-tutorials-master/data/heartregression.csv'
DATA_PATH = '/home/aneesh/Desktop/UTS Literature Survey/Deep Learning Software/Sourcecodes/Datasets/housing.data.csv'
BATCH_SIZE = 2
N_FEATURES = 8

import numpy
import pandas

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# def qmeregression(InFile):
#     data = sio.loadmat(InFile)
#     X = data['X']
#     
#     X_normal = X[:,:,0:50]
#     X_disturb = X[:,:,50:100]
#     
#     return 

# def batch_generator(filenames):
#     filename_queue = tf.train.string_input_producer(filenames)
#     reader = tf.TextLineReader(skip_header_lines=1)
#     _, value = reader.read(filename_queue)
#     
#     # record_defaults are the default values in case some of our columns are empty
#     # This is also to tell tensorflow the format of our data (the type of the decode result)
#     # for this dataset, out of 9 feature columns, 
#     # 8 of them are floats (some are integers, but to make our features homogenous, 
#     # we consider them floats), and 1 is string (at position 5)
#     # the last column corresponds to the lable is an integer
#     record_defaults = [[1.0] for _ in range(N_FEATURES)]
#     record_defaults.append([1])
#     
#     content = tf.decode_csv(value, record_defaults=record_defaults) 
#     features = tf.stack(content[:N_FEATURES])
#     label = content[-1]
# 
#     min_after_dequeue = 10 * BATCH_SIZE
#     capacity = 20 * BATCH_SIZE
#     
#     data_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=BATCH_SIZE, 
#                                         capacity=capacity, min_after_dequeue=min_after_dequeue)
#     
#     return data_batch, label_batch
# 
# def generate_batches(data_batch, label_batch):
#     with tf.Session() as sess:
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#         
#         for _ in range(10): # generate 10 batches
#             features, labels = sess.run([data_batch, label_batch])
#             print(features)
#         
#         coord.request_stop()
# 
#         coord.join(threads)




def main():
#     data_batch, label_batch = batch_generator([DATA_PATH])
#     generate_batches(data_batch, label_batch)
    
    

    
# if __name__ == '__main__':
#     InFile = '/home/aneesh/Documents/UTS-HDU Summer School/summer_project-master/shaokai_jiashen/GrangerCausality/Data/SimulationData.mat'
    
    