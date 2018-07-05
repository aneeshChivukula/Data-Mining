import numpy

import os
import urllib
import gzip
import cPickle as pickle

classlabels = [4,9]
roundingplaces = 2

def mnist_generator(data, batch_size, n_labelled, limit=None):
    images, targets = data

    # sel_set = numpy.logical_or(targets == classlabels[0], targets == classlabels[1])
    # images = images[sel_set]
    # targets = targets[sel_set]
    # images = images[:int(round(images.shape[0], -roundingplaces))]
    # targets = targets[:int(round(targets.shape[0], -roundingplaces))]

    rng_state = numpy.random.get_state()
    numpy.random.shuffle(images)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(targets)
    if limit is not None:
        print "WARNING ONLY FIRST {} MNIST DIGITS".format(limit)
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]
    if n_labelled is not None:
        labelled = numpy.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch():
        rng_state = numpy.random.get_state()
        # print('before images.shape', images.shape)
        # print('before targets.shape', targets.shape)

        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)

        if n_labelled is not None:
            numpy.random.set_state(rng_state)
            numpy.random.shuffle(labelled)

        image_batches = images.reshape(-1, batch_size, 784)
        target_batches = targets.reshape(-1, batch_size)
        # print('after image_batches.shape', image_batches.shape)
        # print('after target_batches.shape', target_batches.shape)
        # import sys
        # sys.exit()





        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in xrange(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]), numpy.copy(labelled))

        else:

            for i in xrange(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]))

    return get_epoch

def load(batch_size, test_batch_size, n_labelled=None):
    filepath = '/tmp/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print "Couldn't find MNIST dataset in /tmp, downloading..."
        urllib.urlretrieve(url, filepath)

    with gzip.open('/tmp/mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)
        # train_data_tup, dev_data_tup, test_data_tup = pickle.load(f)



    # train_sel_set = numpy.logical_or(train_data_tup[1] == classlabels[0], train_data_tup[1] == classlabels[1])
    # train_data = (train_data_tup[0][train_sel_set],train_data_tup[1][train_sel_set])
    #
    # dev_sel_set = numpy.logical_or(dev_data_tup[1] == classlabels[0], dev_data_tup[1] == classlabels[1])
    # dev_data = (dev_data_tup[0][dev_sel_set],dev_data_tup[1][dev_sel_set])
    #
    # test_sel_set = numpy.logical_or(test_data_tup[1] == classlabels[0], test_data_tup[1] == classlabels[1])
    # test_data = (test_data_tup[0][test_sel_set],test_data_tup[1][test_sel_set])
    #
    # print('train_data',train_data)
    # print('dev_data',dev_data)
    # print('test_data',test_data)


    return (
        mnist_generator(train_data, batch_size, n_labelled), 
        mnist_generator(dev_data, test_batch_size, n_labelled), 
        mnist_generator(test_data, test_batch_size, n_labelled)
    )
