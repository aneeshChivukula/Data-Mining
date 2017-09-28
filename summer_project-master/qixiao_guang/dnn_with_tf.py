# -*-coidng:utf-8-*-
# some references:
# http://blog.aloni.org/posts/backprop-with-tensorflow/
# https://pythonprogramming.net/train-test-tensorflow-deep-learning-tutorial/
# https://www.tensorflow.org/api_guides/python/reading_data

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_name, feature_number= "./data/sampleX1", 10
# file_name, feature_number = "./data/sampleX2", 14
# file_name, feature_number = "./data/sampleX3", 13

n_nodes_hl1 = 200
n_nodes_hl2 = 200
n_nodes_hl3 = 200
n_nodes_input = feature_number  # in other words, feature numbers. change file_name also change the variable
n_classes = 1   # labels in classification, values in regression
batch_size = 100

x = tf.placeholder(tf.float32, [None, n_nodes_input])
y = tf.placeholder(tf.float32)


def read_from_csv():
    column_size = n_nodes_input + 1

    df = pd.read_csv(file_name, header=0, index_col=0)
    train_X = np.array(df.iloc[10: 4510, [i for i in range(column_size) if i != 10]].values, dtype=np.float)
    train_Y = np.array(df.iloc[10: 4510, 10].values, dtype=np.float)
    train_Y = train_Y.reshape(train_Y.shape[0], 1)

    test_X = np.array(df.iloc[4510:, [i for i in range(column_size) if i != 10]].values, dtype=np.float)
    test_Y = np.array(df.iloc[4510:, 10].values, dtype=np.float)
    test_Y = test_Y.reshape(test_Y.shape[0], 1)

    return train_X, train_Y, test_X, test_Y


def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_input, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.reduce_mean(l3)  # regression

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    cost = tf.reduce_mean(tf.pow(y-prediction, 2))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(5).minimize(cost)

    hm_epochs = 300
    # the iteration time
    # for sampleX1, 250 times.
    # for sampleX2, 500 times. when the times of epoch are more than 300 times or so,
    # the curve becomes very unstable, but tending to decline
    # for sampleX3, 280 times. very unstable, sometimes 300 times are good,
    # but sometimes awful...i tried 600 after 300 or so becoming unsatble

    record_loss = []  # record loss for drawing
    record_predict_value = []  # record predict training values for drawing
    record_predict_test_value = [] # record predict testing values for drawing

    # saver = tf.train.Saver()

    train_X, train_Y, test_X, test_Y = read_from_csv()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0  # record loss for every epoch
            i = 0
            while i < len(train_X):
                start = i
                end = i + batch_size
                batch_x = train_X[start: end]
                batch_y = train_Y[start: end]
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c  # sum every batch loss
                i += batch_size

            record_loss.append(epoch_loss)

            print 'Epoch', epoch+1, 'completed out of', hm_epochs, 'loss', epoch_loss

        correct = tf.less_equal(tf.abs(prediction - y), 0.1)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        # calculate training accuracy
        for i in range(train_X.shape[0]):
            record_predict_value.append(sess.run(prediction, feed_dict={x: train_X[i:i+1], y: train_Y[i: i+1]}))
        print 'train accuracy:', accuracy.eval({x: train_X, y: train_Y})

        # calculate testing accuracy
        for i in range(test_X.shape[0]):
            record_predict_test_value.append(sess.run(prediction, feed_dict={x: test_X[i:i + 1], y: test_Y[i: i + 1]}))
        print 'test accuracy:', accuracy.eval({x: test_X, y: test_Y})

        # draw loss curve
        plt.figure("loss")
        plt.ylim((0, 150))
        plt.plot(range(hm_epochs), record_loss)

        # draw training fitting curve
        plt.figure("train fit")
        plt.ylim((10, 50))
        plt.plot(range(1000), record_predict_value[3000:4000], 'r')  # 'r' means the line is red
        plt.plot(range(1000), train_Y[3000:4000], 'b')  # 'b' means the is blue

        # draw testing fitting curve
        plt.figure("test fit")
        plt.ylim((30, 50))
        plt.plot(range(len(test_Y)), record_predict_test_value, 'r')
        plt.plot(range(len(test_Y)), test_Y, 'b')

        plt.show()
        # saver.save(sess, "./logs/1/model.cpkt")


if __name__ == '__main__':
   train_neural_network(x)