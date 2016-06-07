import os
import pandas
import random
import sklearn
from sklearn import datasets, cross_validation, metrics

from sklearn.cross_validation import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics.classification import accuracy_score
import tensorflow as tf
from tensorflow.contrib import learn as skflow


matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1, matrix2)
sess = tf.Session()

result = sess.run(product)
print(result)
sess.close()

with tf.Session() as sess:
    result = sess.run([product])
    print(result)

os.chdir('/home/aneesh/tf_examples-master')
data = pandas.read_csv('data/titanic_train.csv')
print(data.shape)
print(data.columns)
y, X = data['Survived'], data[['Age', 'SibSp', 'Fare']].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print accuracy_score(y_test, lr.predict(X_test))

random.seed(42)
digits = datasets.load_digits()
X = digits.images
y = digits.target
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
    test_size=0.2, random_state=42)

def conv_model(X, y):
    X = tf.expand_dims(X, 3)
    features = tf.reduce_max(skflow.ops.conv2d(X, 12, [3, 3]), [1, 2])
    features = tf.reshape(features, [-1, 12])
    return skflow.models.logistic_regression(features, y)

classifier = skflow.TensorFlowEstimator(model_fn=conv_model, n_classes=10,
                                        steps=500, learning_rate=0.05,
                                        batch_size=128)
# classifier = skflow.TensorFlowLinearClassifier(n_classes=2, batch_size=128, steps=500, learning_rate=0.05)
classifier.fit(X_train, y_train)
score = metrics.accuracy_score(classifier.predict(X_test), y_test)
print('Accuracy: %f' % score)




# if __name__ == '__main__':
#     print ''

'''
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
To Do
DBN in DL4J and R

with tf.Session() as sess:
  with tf.device("/gpu:1"):
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])
    product = tf.matmul(matrix1, matrix2)
    ...

What is the difference between a neural network and a deep belief network?
http://stats.stackexchange.com/questions/51273/what-is-the-difference-between-a-neural-network-and-a-deep-belief-network

'''