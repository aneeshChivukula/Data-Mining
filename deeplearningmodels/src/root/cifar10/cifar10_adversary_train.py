from __future__ import division

from PIL import Image
import copy
from datetime import datetime
from deap import base
from deap import creator
from deap import tools
import math
from os import listdir
import os.path
import random
import sys
import time

import numpy as np
from root.cifar10 import cifar10
from root.cifar10 import cifar10_eval
import tensorflow as tf
import time


FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_adversary_train',
#                            """Directory where to write event logs """
#                            """and checkpoint.""")
tf.app.flags.DEFINE_string('adv_train_dir', '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_adversary_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('adv_max_steps', 20,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('adv_log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('adv_eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_integer('low', -255,
                            """Lower limit for pixel value.""")
tf.app.flags.DEFINE_integer('high', 255,
                            """Upper limit for pixel value.""")
tf.app.flags.DEFINE_integer('dividend', 1,
                            """Factor to control the GA norm initialization boundaries.""")
# tf.app.flags.DEFINE_integer('steplow', -50,
#                             """Small step limit for mutation operator.""")
# tf.app.flags.DEFINE_integer('stephigh', 50,
#                             """Small step limit for mutation operator.""")

tf.app.flags.DEFINE_integer('steplow', -5,
                            """Small step limit for mutation operator.""")
tf.app.flags.DEFINE_integer('stephigh', +5,
                            """Small step limit for mutation operator.""")
# tf.app.flags.DEFINE_integer('steplow', -10,
#                             """Small step limit for mutation operator.""")
# tf.app.flags.DEFINE_integer('stephigh', 10,
#                             """Small step limit for mutation operator.""")
# tf.app.flags.DEFINE_integer('steplow', -20,
#                             """Small step limit for mutation operator.""")
# tf.app.flags.DEFINE_integer('stephigh', 20,
#                             """Small step limit for mutation operator.""")
# tf.app.flags.DEFINE_integer('steplow', -100,
#                             """Small step limit for mutation operator.""")
# tf.app.flags.DEFINE_integer('stephigh', 100,
#                             """Small step limit for mutation operator.""")



tf.app.flags.DEFINE_integer('max_iter_test', 100,
                            """Set max_iter to get sufficient mix of positive and negative classes in testing CNN and training GA.""")
# tf.app.flags.DEFINE_integer('numalphas', 2,
#                             """Number of search solutions in the GA algorithm.""")
# tf.app.flags.DEFINE_integer('numalphas', 10,
#                             """Number of search solutions in the GA algorithm.""")
# tf.app.flags.DEFINE_integer('numalphas', 20,
#                             """Number of search solutions in the GA algorithm.""")
# Next experiment : #
# tf.app.flags.DEFINE_integer('numalphas', 50,
#                             """Number of search solutions in the GA algorithm.""")
tf.app.flags.DEFINE_integer('numalphas', 100,
                            """Number of search solutions in the GA algorithm.""")
# tf.app.flags.DEFINE_integer('numgens', 10,
#                             """Number of generations in the GA algorithm.""")
tf.app.flags.DEFINE_integer('numgens', 20,
                            """Number of generations in the GA algorithm.""")
# tf.app.flags.DEFINE_integer('numgens', 50,
#                             """Number of generations in the GA algorithm.""")
# tf.app.flags.DEFINE_integer('numgens', 100,
#                             """Number of generations in the GA algorithm.""")
tf.app.flags.DEFINE_integer('myepsilon', 0.0001,
                            """Parameter determining game iterations.""")
# tf.app.flags.DEFINE_integer('myepsilon', 0.001,
#                             """Parameter determining game iterations.""")
# tf.app.flags.DEFINE_integer('mylambda', 20,
#                             """Parameter determining weight of the error term in fitness function.""")
tf.app.flags.DEFINE_integer('mylambda', 10, # Use this for mylambda
                            """Parameter determining weight of the error term in fitness function.""")
# tf.app.flags.DEFINE_integer('mylambda', 5,
#                             """Parameter determining weight of the error term in fitness function.""")
# tf.app.flags.DEFINE_integer('mylambda', 1,
#                             """Parameter determining weight of the error term in fitness function.""")
# tf.app.flags.DEFINE_integer('mylambda', 0.5,
#                             """Parameter determining weight of the error term in fitness function.""")
# tf.app.flags.DEFINE_integer('mylambda', 0.2,
#                             """Parameter determining weight of the error term in fitness function.""")
# tf.app.flags.DEFINE_integer('mylambda', 0.1,
#                             """Parameter determining weight of the error term in fitness function.""")
# tf.app.flags.DEFINE_integer('mylambda', 0.01,
#                             """Parameter determining weight of the error term in fitness function.""")
# tf.app.flags.DEFINE_integer('perfmetric', 'precision',
#                             """Parameter determining error term in fitness function.""")
# tf.app.flags.DEFINE_integer('perfmetric', "recall",
#                             """Parameter determining error term in fitness function.""")
# tf.app.flags.DEFINE_integer('perfmetric', 'f1score',
#                             """Parameter determining error term in fitness function.""")
# tf.app.flags.DEFINE_integer('perfmetric', 'tpr',
#                             """Parameter determining error term in fitness function.""")
# tf.app.flags.DEFINE_integer('perfmetric', 'fpr',
#                             """Parameter determining error term in fitness function.""")

length = 3073
# perfmetric = "precision"
perfmetric = "recall"
# perfmetric = "f1score"
# perfmetric = "tpr"
# perfmetric = "fpr"
current_milli_time = lambda: int(round(time.time()))
StdoutFile = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_train/StdoutGeneticOps.txt'
fp = open(StdoutFile,'w')
fp.close()


def train(total_loss, global_step):
    # Put following code in a function that is called until convergence with all the train parameters : global_step
    
    num_batches_per_epoch = cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)
    
    lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                              global_step,
                              decay_steps,
                              cifar10.LEARNING_RATE_DECAY_FACTOR,
                              staircase=True)
    
    
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    
    loss_averages_op = loss_averages.apply([total_loss])
    
    
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    
    return train_op


        
def loss_function_input(images,flag):
    kernel1 = tf.convert_to_tensor(np.load(os.path.join(FLAGS.out_dir, 'conv1-weights.npy')))
    conv2d1 = tf.nn.conv2d(images, kernel1, [1, 1, 1, 1], padding='SAME')
#         conv2d1 = tf.nn.conv2d(imagesnew, kernel1, [1, 1, 1, 1], padding='SAME')
    biases1 = tf.convert_to_tensor(np.load(os.path.join(FLAGS.out_dir, 'conv1-biases.npy')))
    bias1 = tf.nn.bias_add(conv2d1, biases1)
    conv1 = tf.nn.relu(bias1)
    
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                     padding='SAME', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                name='norm1')
    
    kernel2 = tf.convert_to_tensor(np.load(os.path.join(FLAGS.out_dir, 'conv2-weights.npy')))
    conv2d2 = tf.nn.conv2d(norm1, kernel2, [1, 1, 1, 1], padding='SAME')
    biases2 = tf.convert_to_tensor(np.load(os.path.join(FLAGS.out_dir, 'conv2-biases.npy')))
    bias2 = tf.nn.bias_add(conv2d2, biases2)
    conv2 = tf.nn.relu(bias2)

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')


    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])

    weights3 = tf.convert_to_tensor(np.load(os.path.join(FLAGS.out_dir, 'local3-weights.npy')))
    biases3 = tf.convert_to_tensor(np.load(os.path.join(FLAGS.out_dir, 'local3-biases.npy')))
    local3 = tf.nn.relu(tf.matmul(reshape, weights3) + biases3)

    weights4 = tf.convert_to_tensor(np.load(os.path.join(FLAGS.out_dir, 'local4-weights.npy')))
    biases4 = tf.convert_to_tensor(np.load(os.path.join(FLAGS.out_dir, 'local4-biases.npy')))
    
    softmaxweights = tf.convert_to_tensor(np.load(os.path.join(FLAGS.out_dir, 'softmax_linear-weights.npy')))
    softmaxbiases = tf.convert_to_tensor(np.load(os.path.join(FLAGS.out_dir, 'softmax_linear-biases.npy')))
    if(flag=='train'):
        local4old = tf.nn.relu(tf.matmul(local3, weights4) + biases4 )
        alpha = cifar10._variable_on_cpu('alpha', [192], tf.constant_initializer(0.1))
        local4 = tf.add(local4old,alpha)
        softmax_linear = tf.add(tf.matmul(local4, softmaxweights), softmaxbiases)
        return (softmax_linear,alpha)
    elif(flag=='test'):
        local4 = tf.nn.relu(tf.matmul(local3, weights4) + biases4 )
        softmax_linear = tf.add(tf.matmul(local4, softmaxweights), softmaxbiases)
        return softmax_linear
    else:
        raise ValueError('Please supply a input directory')
        
    
            
        
def adversary_test_cnn():
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.adv_log_device_placement))
        coord = tf.train.Coordinator()

#         global_step = tf.Variable(0, trainable=False)
        
        eval_data = FLAGS.adv_eval_data == 'test'
        
        images, labels = cifar10.inputs(eval_data=eval_data)
        
        softmax_linear = loss_function_input(images,'test')
        labels = tf.cast(labels, tf.int64)
        
        top_k_op = tf.nn.in_top_k(softmax_linear, labels, 1)
        
        init = tf.initialize_all_variables()
        sess.run(init)
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
#             num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            num_iter = FLAGS.max_iter_test
#             true_count = 0
            true_positives_count = 0
            false_positives_count = 0
            true_negatives_count = 0
            false_negatives_count = 0
            perfmetrics = {}

#             total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            
            while step < num_iter and not coord.should_stop():
#                 predictions = sess.run([top_k_op])

                is_label_one = sess.run(labels).astype(bool)
                is_label_zero = np.logical_not(is_label_one)
                
                correct_prediction = sess.run([top_k_op])
                false_prediction = np.logical_not(correct_prediction)
                
                true_positives_count += np.sum(np.logical_and(correct_prediction, is_label_one))
                false_positives_count += np.sum(np.logical_and(false_prediction, is_label_zero))
                 
                true_negatives_count += np.sum(np.logical_and(correct_prediction, is_label_zero))
                false_negatives_count += np.sum(np.logical_and(false_prediction, is_label_one))     
                
#                 true_positives_count += np.sum(np.logical_and(correct_prediction, is_label_one))
#                 false_positives_count += np.sum(np.logical_and(correct_prediction, is_label_zero))
#  
#                 true_negatives_count += np.sum(np.logical_and(false_prediction, is_label_one))
#                 false_negatives_count += np.sum(np.logical_and(false_prediction, is_label_zero))     
#                 print('labels',sess.run(labels))
                
#                 true_count += np.sum(predictions)
                step += 1
#                 print('is_label_one',is_label_one)
#                 print('sess.run(softmax_linear)',sess.run(softmax_linear))
#                 print('predictions',predictions)
#                 print('labels',labels)
#             sys.exit()            
            print('true_positives_count',true_positives_count)
            print('false_positives_count',false_positives_count)
            print('false_negatives_count',false_negatives_count)
            print('true_negatives_count',true_negatives_count)
            
            precision = float(true_positives_count) / float(true_positives_count+false_positives_count)
            recall = float(true_positives_count) / float(true_positives_count+false_negatives_count)
            f1score = 2*float(true_positives_count) / (2*float(true_positives_count)+float(false_positives_count + false_negatives_count))
            tpr = float(true_positives_count) / float(true_positives_count+false_negatives_count)
            fpr = float(false_positives_count) / float(false_positives_count+true_negatives_count)

            perfmetrics['precision'] = precision
            perfmetrics['recall'] = recall
            perfmetrics['f1score'] = f1score
            perfmetrics['tpr'] = tpr
            perfmetrics['fpr'] = fpr

            print('precision',precision)
            print('recall',recall)
            print('f1score',f1score)
            
#             precision = (true_count / total_sample_count)
#             print('%s: adversary_test_cnn precision @ 1 = %.3f' % (datetime.now(), precision))
        except Exception as e:  
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
        
        print('%s: adversary training error @ 1 = %.3f' % (datetime.now(), FLAGS.mylambda * perfmetrics[str(perfmetric)]))
        
#         return(1-precision)
        return perfmetrics
        
        
def adversary_train_cnn():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        
        images, labels = cifar10.distorted_inputs()

#         alpha = cifar10._variable_on_cpu('alpha', [24, 24, 3], tf.constant_initializer(0.1))
#         imagesnew = tf.add(images,alpha)
        softmax_linear,alpha = loss_function_input(images,'train')
        
        
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(softmax_linear, labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')        
        
        loss = cross_entropy_mean - tf.nn.l2_loss(alpha) 
        train_op = train(loss, global_step)

        init = tf.initialize_all_variables()
        
        sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.adv_log_device_placement))
        
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        
        for step in xrange(FLAGS.adv_max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time
            
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            
            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))
                
#                 print('alpha',sess.run(alpha))
                
        
#         print((sess.run(tf.add(images,alpha))).shape)
        
        
        np.save(os.path.join(FLAGS.out_dir, 'alpha.npy'), sess.run(alpha))
        
        

        

        # Write f(alpha*W+(Wx+b))
        # Complete cifar10_train : inference, loss, train, session

        
#         print(tf.truncated_normal_initializer(stddev=1e-4))
#         print(tf.convert_to_tensor(np.load(os.path.join(FLAGS.out_dir, 'conv1-weights.npy'))))
#         tf.convert_to_tensor(np.load(os.path.join(FLAGS.out_dir, 'conv1-weights.npy'))

#         with tf.variable_scope('conv1') as scope:
#             kernel = np.load(os.path.join(FLAGS.out_dir, 'conv1-weights.npy')) 
#             conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
#             biases = np.load(os.path.join(FLAGS.out_dir, 'conv1-biases.npy'))
#             bias = tf.nn.bias_add(conv, biases)
#             conv1 = tf.nn.relu(bias, name=scope.name)
#         
#         pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
#                          padding='SAME', name='pool1')
#         norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
#                     name='norm1')
        
        


def initIndividualImage(filename):
#     return icls(Image.open(filename).getdata()).reshape((32,32,3))
    img = Image.open(filename)
    img.load()
    return np.asarray( img, dtype="int32" )
#     return np.array(Image.open(filename).getdata()).reshape((32,32,3))


# def initIndividual(icls):
def initIndividual(meanimage):
#     return meanimage + np.random.randint(low=FLAGS.low,high=FLAGS.high, size=(32, 32, 3))
#     return meanimage + np.random.randint(low=FLAGS.steplow,high=FLAGS.stephigh, size=(32, 32, 3))
    return meanimage + np.random.randint(low=random.randint(math.floor(FLAGS.low/FLAGS.dividend),0),high=random.randint(0,math.ceil(FLAGS.high/FLAGS.dividend)), size=(32, 32, 3))

def initImagePopulation(ind_init, InDir):
    images = list()
#     ind = 0
    ls = listdir(InDir)
    ls.sort()
    positiveimagesmean = np.zeros((32, 32, 3))
#     d = ls[0]
    dr = 0
    for d in ls:
        ind = ls.index(d)
        for f in listdir(InDir + d):
            a = ind_init(filename=InDir + d + '/' + f)
            if(len(a.shape) == 3):
                images.append((ind,a))
                if(ind==1):
                    positiveimagesmean = positiveimagesmean + a
                    dr = dr + 1
#         ind = ind + 1
#     print('l',l)
    return images,np.floor(np.divide(positiveimagesmean, dr))

def evaluate(currpopulation):
    binarizer(FLAGS.data_dir + '/imagenet2010-batches-bin/',currpopulation,'test.bin')
    return adversary_test_cnn()

def select(population):
    popsize = len(population)
    popindices = range(0,popsize)

    fitnesses = []
    for p in population:
        fitnesses.append(p.fitness.weights[0])
    
#     fitnesses = [0.25,0.5,0.15,0.1]    
#     pr = np.round(np.divide(fitnesses,max(fitnesses)),2)
#     if(sum(np.round(np.divide(fitnesses,sum(fitnesses)),2)) == 1):
#         pr = np.round(np.divide(fitnesses,max(fitnesses)),2)
#     else:
#         pr = np.round(np.divide(np.subtract(fitnesses, min(fitnesses)),max(fitnesses) - min(fitnesses)),2)
#     print('np.round(np.divide(fitnesses,sum(fitnesses)),2)',np.round(np.divide(fitnesses,sum(fitnesses)),2))
    
    randompopindices = np.random.choice(a=popindices,size=int(popsize/2),replace=False,p= fitnesses / np.linalg.norm(fitnesses,1))
    
#     print('popindices',popindices)
#     print('randompopindices',randompopindices)
#     print('fitnesses / np.linalg.norm(fitnesses,1)',fitnesses / np.linalg.norm(fitnesses,1))
#     print('fitnesses',fitnesses)
#     print('population',population)
#     sys.exit()
    
    
#     L = [population[i] for i in randompopindices]
#     return ([population[i] for i in randompopindices],[population[i] for i in popindices if i not in randompopindices])
    return [population[i] for i in randompopindices]
#     return L

def mutation(individual):
    mask = np.random.randint(0,2,size=(32, 32, 3)).astype(np.bool)
    r = np.full((32, 32, 3),random.randint(FLAGS.steplow,FLAGS.stephigh))
#     print('individual[0] before',np.sum(individual[0]))
#     print('individual[0] shape before',(individual[0]).shape)
    individual[0][mask] = individual[0][mask] + r[mask]
#     print('individual[0] after',np.sum(individual[0]))
#     print('individual[0] shape after',(individual[0]).shape)
    return individual

def crossover(individual1,individual2):
    
    np.random.seed(current_milli_time())

    heightstartind = np.random.randint(low=0,high=np.random.randint(1,16))
    heightendind = np.random.randint(heightstartind + np.random.randint(2,10),32)

    widthstartind = np.random.randint(low=0,high=np.random.randint(1,16))
    widthendind = np.random.randint(widthstartind + np.random.randint(2,10),32)
    
    before1 = np.copy(individual1[0])
    before2 = np.copy(individual2[0])
#     print('individual1 before',individual1[0])
#     print('individual2 before',individual2[0])
    
    individual1[0][heightstartind:heightendind,widthstartind:widthendind,], individual2[0][heightstartind:heightendind,widthstartind:widthendind,] = individual2[0][heightstartind:heightendind,widthstartind:widthendind,].copy(), (individual1[0][heightstartind:heightendind,widthstartind:widthendind,]).copy()
    
    after1 = np.copy(individual1[0])
    after2 = np.copy(individual2[0])
#     print('individual1 after',individual1[0])
#     print('individual2 after',individual2[0])
    
#     print('np.count_nonzero(after1-before1) == 0',np.count_nonzero(after1-before1) == 0)
#     print('np.count_nonzero(after2-before2) == 0',np.count_nonzero(after2-before2) == 0)

    if((np.count_nonzero(after1-before1) == 0 or np.count_nonzero(after2-before2) == 0)):
       fp = open(StdoutFile,'a')
       print('Skipped a mutation. Check for logical bugs.')
       print('np.count_nonzero(after1-before1) == 0',np.count_nonzero(after1-before1) == 0)
       print('np.count_nonzero(after2-before2) == 0',np.count_nonzero(after2-before2) == 0)
       fp.write('Skipped a mutation. Check for logical bugs.' + '\n')
       fp.write('np.count_nonzero(after1-before1) == 0',np.count_nonzero(after1-before1) == 0 + '\n')
       fp.write('np.count_nonzero(after2-before2) == 0',np.count_nonzero(after2-before2) == 0 + '\n')
#        fp.write('before1',before1 + '\n')
#        fp.write('before2',before2 + '\n')
#        fp.write('after1',after1 + '\n')
#        fp.write('after2',after2 + '\n')
       fp.write('heightstartind',heightstartind + '\n')
       fp.write('heightendind',heightendind + '\n')
       fp.write('widthstartind',widthstartind + '\n')
       fp.write('widthendind',widthendind + '\n')
       fp.close()
#        sys.exit() 
    
    return (individual1, individual2)


def binarizer(CurrDir,population,OutFile):
    
    os.chdir(CurrDir)
    binfile = open(OutFile, 'wb',)
    
    L = []
    for t in population:
        l = np.insert(t[0].flatten(order='F'),0, t[1])
        if(len(l) == length):
            L.append(l)
    np.concatenate(L).astype('int16').tofile(binfile)
    binfile.close()

def tensornorm(curralpha):    
#     return (np.sqrt(np.sum(np.square(curralpha)))) # Divide by avg of l2 norm of training data to get within range of 0,1
    return float(np.sqrt(np.sum(np.square(curralpha))/(32*32*3)) / 255)
#     return float(np.sqrt(np.sum(np.square(curralpha))/(32*32*3*256)))

def distorted_image(x,curralpha):
    a = (curralpha + x)
    a[a>255] = 255
    a[a<0] = 0
    return a

def alphasfitnesses(alphaspopulation,imagespopulation,toolbox):
    fitnesses = []
#     fitnesses = np.zeros(len(alphaspopulation))
    
#     alphanorms = []
#     totnorm = 0.0
#     for index,curralpha in enumerate(alphaspopulation):
#         alphanorms.append(tensornorm(curralpha))
#         totnorm = totnorm + alphanorms[index]
#         alphanorms[index] = tensornorm(curralpha)
    
#     print('alphaspopulation',alphaspopulation)
    
    for index,curralpha in enumerate(alphaspopulation):
        distortedimages = []
        for x in imagespopulation:
            distortedimages.append((distorted_image(x[1],curralpha),x[0]))
#             print('x[0]',x[0])
#         np.append(fitnesses,1 + toolbox.evaluate(distortedimages) - (alphanorms[index]/totnorm))
        print('Reset fitnesses in alphasfitnesses')
#         print('distortedimages',distortedimages)
        
                
        perfmetrics = toolbox.evaluate(distortedimages)
        error = FLAGS.mylambda * (1-perfmetrics[str(perfmetric)])
        fit = 1 + error - tensornorm(curralpha)
        fitnesses.append(fit)
#         fitnesses.append(1 + error - (alphanorms[index]/totnorm))
        alphaspopulation[index].fitness.error = error
#         alphaspopulation[index].fitness.payoff = fit
        alphaspopulation[index].fitness.weights = (fit,)
        alphaspopulation[index].fitness.values = [fit]

        alphaspopulation[index].fitness.precision = perfmetrics['precision']
        alphaspopulation[index].fitness.recall = perfmetrics['recall']
        alphaspopulation[index].fitness.f1score = perfmetrics['f1score']
        alphaspopulation[index].fitness.tpr = perfmetrics['tpr']
        alphaspopulation[index].fitness.fpr = perfmetrics['fpr']
        
        print('Reset fitnesses in alphasfitnesses')
        
        print('error in alphasfitnesses',error)
        print('tensornorm(curralpha) in alphasfitnesses',tensornorm(curralpha))


    print('len(alphaspopulation) in alphasfitnesses',len(alphaspopulation))
    print('fit in alphasfitnesses',fit)

#     return fitnesses / sum(fitnesses)
#     return np.divide(fitnesses, np.sum(fitnesses))


def copyindividuals(offspring,toolbox):
    indcs = []
    for ind in offspring:
        indc = toolbox.clone(ind)
        fit = ind.fitness.weights[0]
        indc.fitness.weights = (fit,)
        indc.fitness.values = [fit]
        indc.fitness.error = ind.fitness.error
        indc.fitness.precision = ind.fitness.precision
        indc.fitness.recall = ind.fitness.recall
        indc.fitness.f1score = ind.fitness.f1score
        indc.fitness.tpr = ind.fitness.tpr
        indc.fitness.fpr = ind.fitness.fpr
#         indc.fitness.payoff = ind.fitness.payoff
        indcs.append(indc)
    return indcs

def adversary_train_genetic(InDir,WeightsDir):

#     creator.create("FitnessMax", base.Fitness, weights=(0.0,),precision=0.0,payoff=0.0)
    creator.create("FitnessMax", base.Fitness, weights=(0.0,),error=0.0,precision=0.0,recall=0.0,f1score=0.0,tpr=0.0,fpr=0.0)
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    toolbox.register("mutate", mutation)
    toolbox.register("mate", crossover)
    toolbox.register("evaluate", evaluate)
    toolbox.register("select", select)
    
    toolbox.register("individualImage", initIndividualImage)
    toolbox.register("imagepopulation", initImagePopulation, toolbox.individualImage)
    imagespopulation,positiveimagesmean = toolbox.imagepopulation(InDir)

    toolbox.register("attribute",initIndividual, meanimage=positiveimagesmean)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=FLAGS.numalphas)
    alphaspopulation = toolbox.population()

    alphasfitnesses(alphaspopulation,imagespopulation,toolbox)
    print('Calling alphasfitnesses before')
    print('len(alphaspopulation)',len(alphaspopulation))
#     print('alphaspopulation before',(alphaspopulation))
#     print('len(alphaspopulation) before',len(alphaspopulation))

#     CXPB, MUTPB, NGEN = 0.5, 0.2, 40
    gen = 0
#     exitLoop = False
#     while (gen < (NGEN) and not exitLoop):
    while (gen < FLAGS.numgens):
        print('gen',gen)
        
        popsumbefore = np.sum(alphaspopulation)

        selectedoffspring = toolbox.select(alphaspopulation)
        
        parents = copyindividuals(selectedoffspring,toolbox)
        offspring = copyindividuals(selectedoffspring,toolbox)
#         offspring = []
        
#         offspring = map(toolbox.clone, offspring)
#         parents = map(toolbox.clone, offspring)

        print('Initialization completed')
            
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
#             if random.random() < CXPB:
                print('Calling mate')
                
#                 sumbefore1 = np.sum(child1[0])
#                 child1before = np.copy(offspring[0])
                                
#                 sumbefore2 = np.sum(child2[0])
#                 print('child1 before',child1)

                (child1m,child2m) = toolbox.mate(child1, child2)
                child1[0] = np.copy(child1m[0])
                child2[0] = np.copy(child2m[0])
                child1 = toolbox.clone(child1)
                child2 = toolbox.clone(child2)
#                 (child1, child2) = tollbox.clone(toolbox.mate(child1, child2))


#                 sumafter1 = np.sum(child1[0])
#                 child1after = np.copy(offspring[0])

                
#                 print('child1 after',child1)
#                 print('offspring[0] after',offspring[0])

#                 print('sumbefore1',sumbefore1)
#                 print('sumafter1',sumafter1)
#                 print('child1after - child1before',np.sum(child1after - child1before))

                
#                 print('offspring[0] - child1 after',sum(offspring[0] - child1))
#                 print('offspring[0] - child1m after',sum(offspring[0] - child1m))

#                 sys.exit()


                
#                 (child1,child2) = toolbox.clone(toolbox.mate(child1, child2))

                print('Reset mate weights')
                del child1.fitness.values
                child1.fitness.weights = (0.0,)
                
                del child2.fitness.values
                child2.fitness.weights = (0.0,)
                
                print('child1.fitness.valid',child1.fitness.valid)
                print('child2.fitness.valid',child2.fitness.valid)
                
#                 print('child2.fitness.error',child2.fitness.error)
#                 child1c = toolbox.clone(child1)
#                 child2c = toolbox.clone(child2)
#                 sumafter1 = np.sum(child1[0])
#                 sumafter2 = np.sum(child2[0])
#                 print('sumafter1 - sumbefore1',sumafter1 - sumbefore1)
#                 print('sumafter2 - sumbefore2',sumafter2 - sumbefore2)
#                 offspring.append(child1c)
#                 offspring.append(child2c)
#                 print('np.sum(parents[0])',np.sum(parents[0]))
#                 print('np.sum(child1c)',np.sum(child1c))
#                 sys.exit()
#         print('np.sum(parents[0])',np.sum(parents[0]))
#         print('np.sum(offspring[0])',np.sum(offspring[0]))
# 
#         sys.exit()
                
        for mutant in offspring:
#             if random.random() < MUTPB:
                print('Calling mutate')
#                 print('mutant[0] before',np.sum(mutant[0]))
#                 print('mutant[0] shape before',(mutant[0]).shape)
                
                mutantm = toolbox.mutate(mutant)
                mutant[0] = np.copy(mutantm[0])
                mutant = toolbox.clone(mutant)
#                 mutant = toolbox.clone(toolbox.mutate(mutant))
                
                print('Reset mutant weights')
                del mutant.fitness.values
                mutant.fitness.weights = (0.0,)
                
#                 print('mutant.fitness.valid',mutant.fitness.valid)
#                 print('mutant.fitness.weights',mutant.fitness.weights)
#                 print('mutant[0] after',np.sum(mutant[0]))
#                 print('mutant[0] shape after',(mutant[0]).shape)
        
        print('Calling alphasfitnesses after')
        alphasfitnesses(offspring,imagespopulation,toolbox)
        
#         invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#         print('len(invalid_ind)',len(invalid_ind))
#         print('len(offspring)',len(offspring))
#         print('len(parents)',len(parents))
#         sys.exit()
#         print('len(alphaspopulation)',len(alphaspopulation))
#         print('alphaspopulation after',(alphaspopulation))
#         print('len(alphaspopulation) after',len(alphaspopulation))
#         if(len(invalid_ind) != 0):
#             alphasfitnesses(invalid_ind,imagespopulation,toolbox)
#             fitnesses = []
#             for p in invalid_ind:
#                 print('p.fitness.weights',p.fitness.weights)
#                 print('p.fitness.valid',p.fitness.valid)
#                 fitnesses.append(p.fitness.weights[0])
#             print('reset fitnesses end of curr gen',fitnesses)

        alphaspopulation[:] = copyindividuals(parents + offspring,toolbox)


        popsumafter = np.sum(alphaspopulation)


        print('popsumbefore',popsumbefore)
        print('popsumafter',popsumafter)
#         sys.exit()
#         fitnesses = []
#         for p in parents:
#             fitnesses.append(p.fitness.weights[0])
#         print('fitnesses parents gen',fitnesses)
# 
#         fitnesses = []
#         for p in offspring:
#             fitnesses.append(p.fitness.weights[0])
#         print('fitnesses offspring gen',fitnesses)
#         
#         fitnesses = []
#         for p in alphaspopulation:
#             fitnesses.append(p.fitness.weights[0])
#         print('fitnesses end of curr gen',fitnesses)


#         print('len(alphaspopulation)',len(alphaspopulation))
#         print('len(parents)',len(parents))
#         print('len(offspring)',len(offspring))
#         import sys
#         sys.exit()
        
#         if(len(alphaspopulation) != 0 and len(invalid_ind) != 0):
#             alphasfitnesses(invalid_ind,imagespopulation,toolbox)
#             alphaspopulation[:] = offspring
#         else:
#             exitLoop = True
        gen = gen + 1
        
    
    return (alphaspopulation,imagespopulation)
        



    
    
    # register genetic operators and evaluation method
    # randomized indexing
    # sorted selection
    # parallel processing alphas
    # constant sum game
    # update minutes
    # for big data, implement in tf than numpy
    

# def main(argv=None):
#     if tf.gfile.Exists(FLAGS.adv_train_dir):
#         tf.gfile.DeleteRecursively(FLAGS.adv_train_dir)
#     tf.gfile.MakeDirs(FLAGS.adv_train_dir)
#     adversary_train_cnn()


    
if __name__ == '__main__':
#   tf.app.run()
  InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TrainSplit/' 
  WeightsDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_output'
  (alphaspopulation,imagespopulation) = adversary_train_genetic(InDir,WeightsDir)
  print('final len(alphaspopulation)',len(alphaspopulation))
#   print('input imagespopulation',imagespopulation)
  
  
  