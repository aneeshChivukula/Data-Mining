import os.path
import time

import numpy as np
import tensorflow as tf

from root.cifar10 import cifar10
from datetime import datetime

FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_adversary_train',
#                            """Directory where to write event logs """
#                            """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir', '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_adversary_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 20,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

length = 3073

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
        global_step = tf.Variable(0, trainable=False)
        
        eval_data = FLAGS.eval_data == 'test'
        
        images, labels = cifar10.inputs(eval_data=eval_data)
        
        softmax_linear = loss_function_input(images)
        
        labels = tf.cast(labels, tf.int64)
        softmax_probabilities = tf.nn.softmax(softmax_linear)
#         cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')  
        
        loss = softmax_probabilities
        
        train_op = train(loss, global_step)
        
        init = tf.initialize_all_variables()
        
        sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
        
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

        for step in xrange(FLAGS.max_steps):
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

        
        return(sess.run(loss_value))
        
        
def adversary_train_cnn():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        
        images, labels = cifar10.distorted_inputs()

#         alpha = cifar10._variable_on_cpu('alpha', [24, 24, 3], tf.constant_initializer(0.1))
#         imagesnew = tf.add(images,alpha)
        softmax_linear,alpha = loss_function_input(images)
        
        
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(softmax_linear, labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')        
        
        loss = cross_entropy_mean - tf.nn.l2_loss(alpha) 
        train_op = train(loss, global_step)

        init = tf.initialize_all_variables()
        
        sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
        
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        
        for step in xrange(FLAGS.max_steps):
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
                
                print('alpha',sess.run(alpha))
                
        
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
        
        
from deap import base
from deap import creator
from deap import tools
from PIL import Image
from os import listdir


def initIndividualImage(icls, filename):
#     return icls(Image.open(filename).getdata()).reshape((32,32,3))
    img = Image.open(filename)
    img.load()
    return np.asarray( img, dtype="int32" )
#     return np.array(Image.open(filename).getdata()).reshape((32,32,3))


def initIndividual(icls):
    return np.random.randint(256, size=(32, 32, 3))

def initPopulation(ind_init, InDir):
    l = list()
    ind = 0
    ls = listdir(InDir)
    ls.sort()
    for d in ls:
        for f in listdir(InDir + d):
            a = ind_init(filename=InDir + d + '/' + f)
            if(len(a.shape) == 3):
                l.append((ind,ind_init(filename=InDir + d + '/' + f)))
        ind = ind + 1
#     print('l',l)
    return l


def evaluate(currpopulation,trainclasses):
    binarizer(FLAGS.data_dir + '/imagenet2010-batches-bin/',currpopulation,trainclasses,'test.bin')
    return adversary_test_cnn()

def select(population,popsize,fitnesses):
    return np.random.choice(a=population,size=len(population)/2,replace=True,p=fitnesses)

def mutation(individual):
    mask = np.random.randint(0,2,size=(32, 32, 3)).astype(np.bool)
    r = np.random.randint(256, size=(32, 32, 3))
    individual[mask] = r[mask]
    return (individual,)

def crossover(individual1,individual2):
    heightstartind = np.random.randint(256)
    heightendind = np.random.randint(heightstartind,256)
    
    widthstartind = np.random.randint(256)
    widthendind = np.random.randint(widthstartind,256)
    
    individual2[heightstartind:heightendind,widthstartind:widthendind,], individual1[heightstartind:heightendind,widthstartind:widthendind,] = individual1[heightstartind:heightendind,widthstartind:widthendind,].copy(), individual2[heightstartind:heightendind,widthstartind:widthendind,].copy()

    return (individual1, individual2)

# def selection(individual1,individual2):

def binarizer(CurrDir,population,classes,OutFile):
    
    os.chdir(CurrDir)
    binfile = open(OutFile, 'wb',)
    
    L = []
    for t in population:
        l = np.insert(t[0].flatten(order='F'),0, t[1])
        if(len(l) == length):
            L.append(l)

    print(len(L)*length)
    print(len(np.concatenate(L)))
    
    np.concatenate(L).astype('int16').tofile(binfile)
    binfile.close()


def adversary_train_genetic(InDir,WeightsDir):
    
    creator.create("FitnessMax", base.Fitness, weights=(0.0,))
#     creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("individual", initIndividual, creator.Individual)
    toolbox.register("individualImage", initIndividualImage, creator.Individual)
    ind1 = (toolbox.individualImage(filename=InDir+'BlackDog/n02111277_9983.JPEG'))
#     ind2 = (toolbox.individualImage(filename=InDir+'BlackDog/n02111277_9983.JPEG'))
        
    toolbox.register("imagepopulation", initPopulation, toolbox.individualImage, InDir)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mutation", mutation)
    toolbox.register("crossover", crossover)
    toolbox.register("evaluation", evaluate)
    toolbox.register("selection", select)



    imagepopulation = toolbox.imagepopulation()
    
#     print('imagepopulation',imagepopulation)

    numalphas = 10
    population = toolbox.population(n=numalphas)    
#     print('population',(population)[0])
    
    curralpha = population[0]
    
    currpopulation = map(lambda x: ((curralpha + x[1]),x[0]), imagepopulation)
#     print(currpopulation)
    
    
    
    
    


    import sys
    sys.exit()
    
    np.load()



    
    
    # register genetic operators and evaluation method
    # randomized indexing
    # sorted selection
    # parallel processing alphas
    # constant sum game
    # update minutes
    # for big data, implement in tf than numpy
    

def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    adversary_train_cnn()


    
if __name__ == '__main__':
#   tf.app.run()
  InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TrainSplit/' 
  WeightsDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_output'
  adversary_train_genetic(InDir,WeightsDir)
  
  
  
  