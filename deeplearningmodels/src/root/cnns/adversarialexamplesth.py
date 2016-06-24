import theano
import theano.tensor as T
from theano import pp
import numpy

a = T.dscalar()
b = T.dscalar()
c = a + b
f = theano.function([a,b], c)
assert 4.0 == f(1.5, 2.5)

x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = theano.function([x], s)
print(logistic([[0, 1], [-1, -2]]))

x = T.dmatrix('x')
s = T.sum(1 / (1 + T.exp(-x)))
gs = T.grad(s, x)
dlogistic = theano.function([x], gs)
print(dlogistic([[0, 1], [-1, -2]]))

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np
# import cv2

print "[X] downloading data..."
dataset = datasets.fetch_mldata("MNIST Original")

(trainX, testX, trainY, testY) = train_test_split(
    dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.33)

dbn = DBN(
    [trainX.shape[1], 300, 10],
    learn_rates = 0.3,
    learn_rate_decays = 0.9,
    epochs = 10,
    verbose = 1)
dbn.fit(trainX, trainY)

preds = dbn.predict(testX)
print classification_report(testY, preds)

for i in np.random.choice(np.arange(0, len(testY)), size = (10,)):
    pred = dbn.predict(np.atleast_2d(testX[i]))
    image = (testX[i] * 255).reshape((28, 28)).astype("uint8")
    print "Actual digit is {0}, predicted {1}".format(testY[i], pred[0])
#     cv2.imshow("Digit", image)
#     cv2.waitKey(0)



















'''
Theano, Lasagne, nolearn, keras, pylearn2, plato, crino
Only older version of nolearn has DBN. Newer version of nolearn has DNN.
In yaml, pylearn2 has a DBM, RBM
Standard DBN code is not available in python packages. CNNs are readily available.
We must implement a DBN by adapting existing code. Or must trace the dependencies of github packages providing code(like plato). 
More packages to be checked : pylearn2, Torch7, DL4J, H2O, DeepLearnToolbox, MXNet in yaml, lua, Matlab, Java, R 
Alternatves to nolearn in python are Java packages like DL4J, R packages like deepnet
For using GPUs, DL4J consolidates blas into ND4J that has jdk, cuda, nvidia dependencies for switiching backends on linear algebra operations 
ND4J allows portability of deep learning code.  DL4J pipelines also works on both hadoop and spark shell supporting blas, iterative reduce. So main focus is on tuning network hyperparameters and interpreting output in the deep learning model. 
DL4J has both java and scala api docs for building computation graphs and deep learning models. Canova allows us to vectorize and split data input to neural networks. 
DL4J has detailed tutorials for IntelliJ than Eclipse IDE, Maven than SBT build tool
Thus, DL4J is like the Weka of deep learning
https://github.com/Theano/Theano/wiki/Related-projects
http://scikit-learn.org/stable/modules/neural_networks.html
http://scikit-learn.org/dev/modules/neural_networks_supervised.html
https://pythonhosted.org/nolearn/lasagne.html#module-nolearn.lasagne
https://pythonhosted.org/nolearn/lasagne.html
http://www.slideshare.net/roelofp/python-for-image-understanding-deep-learning-with-convolutional-neural-nets
https://github.com/dnouri/nolearn
http://www.pyimagesearch.com/2014/09/22/getting-started-deep-learning-python/
https://github.com/ottogroup/kaggle/blob/master/Otto_Group_Competition.ipynb
http://derekjanni.github.io/Easy-Neural-Nets/
http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
https://blog.dominodatalab.com/gpu-computing-and-deep-learning/

http://deeplearning.net/software/theano/tutorial/index.html#tutorial
http://deeplearning.net/software/theano/introduction.html#introduction
http://deeplearning.net/software/theano/extending/graphstructures.html
http://deeplearning.net/software/pylearn2/library/index.html
https://wiki.python.org/moin/UsingAssertionsEffectively

https://github.com/petered/plato
https://github.com/petered/plato/wiki/Algorithms-in-Plato
http://blocks.readthedocs.io/en/latest/api/algorithms.html
http://deeplearning.net/software/pylearn2/library/models.html
https://groups.google.com/forum/#!topic/pylearn-dev/cBNms1QEmXc

http://deeplearning4j.org/compare-dl4j-torch7-pylearn.html
http://deeplearning4j.org/deepbeliefnetwork.html
http://deeplearning4j.org/restrictedboltzmannmachine.html
https://github.com/deeplearning4j
http://deeplearning4j.org/
https://www.quora.com/What-are-the-best-packages-for-deep-learning-in-R
'''
