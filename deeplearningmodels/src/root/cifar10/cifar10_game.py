import sys
import tensorflow as tf
import numpy as np
import Image
from shutil import copyfile

from root.cifar10 import cifar10

from root.cifar10 import cifar10_train
from root.cifar10 import cifar10_eval
from root.cifar10 import cifar10_adversary_train
from root.cnns import createdataset

from os import listdir
from deap import base
from deap import creator
from deap import tools

FLAGS = tf.app.flags.FLAGS

# perfmetric = "precision"
# perfmetric = "recall"
perfmetric = "f1score"
# perfmetric = "tpr"
# perfmetric = "fpr"


def binarizer(GameInDir,AdvInDir,imagespopulation,curralpha,labels,infile):
    if tf.gfile.Exists(AdvInDir):
      tf.gfile.DeleteRecursively(AdvInDir)
    tf.gfile.MakeDirs(AdvInDir)
    for Label in labels:
        tf.gfile.MakeDirs(AdvInDir + Label)

    binfile = open(GameInDir + infile, 'wb',)
    L = []
#     distortedimages = []
    for i,x in enumerate(imagespopulation):
        CurrLabel = labels[x[0]]
        CurrImage = np.array(cifar10_adversary_train.distorted_image(x[1],curralpha), np.uint8)[0]
        Image.fromarray(CurrImage).save(AdvInDir + CurrLabel + "/" + str(i) + ".jpeg")
        
        l = np.insert(CurrImage.flatten(order='F'),0, x[0])
        if(len(l) == createdataset.length):
            L.append(l)
#         distortedimages.append(CurrImage)
#     print('final distortedimages',distortedimages)
    
    np.concatenate(L).astype('int16').tofile(binfile)
    binfile.close()
    

def main(argv=None):

    WeightsDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_output'
    AdvInDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/AdversarialSplit/'
    GameInDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_data/imagenet2010-batches-bin/'
    EvalDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_eval'
    TrainWeightsDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_train'
    
#     maxiters = 11
    LoopingFlag = True
#     total_iters = 0
    adv_payoff_highest = 0
    gen = 0
    
#     alphastar = np.zeros((32, 32, 3))
    finalresults = []
    

#     InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TestSplit/'
#     imagespopulation,positiveimagesmean = toolbox.imagepopulation(InDir)
#     precision = 1-cifar10_adversary_train.evaluate(imagespopulation)

    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/' 
    createdataset.binarizer(InDir,'TrainSplit/','train.bin')
    copyfile(InDir + 'train.bin', GameInDir + 'train.bin')
    if tf.gfile.Exists(TrainWeightsDir):
      tf.gfile.DeleteRecursively(TrainWeightsDir)
    tf.gfile.MakeDirs(TrainWeightsDir)
    cifar10_train.train()

    createdataset.binarizer(InDir,'TrainSplit/','test.bin')
    copyfile(InDir + 'test.bin', GameInDir + 'test.bin')
    if tf.gfile.Exists(EvalDir):
      tf.gfile.DeleteRecursively(EvalDir)
    tf.gfile.MakeDirs(EvalDir)
    perfmetrics = cifar10_eval.evaluate()
    perf = perfmetrics[str(perfmetric)]
    print('initial original training data performance of cifar10_eval without alphastar on original training data',perf)
    finalresults.append((1, 0, 1, perf, perfmetrics, gen))
    
    createdataset.binarizer(InDir,'TestSplit/','test.bin')
    copyfile(InDir + 'test.bin', GameInDir + 'test.bin')
    if tf.gfile.Exists(EvalDir):
      tf.gfile.DeleteRecursively(EvalDir)
    tf.gfile.MakeDirs(EvalDir)
    perfmetrics = cifar10_eval.evaluate()
    perf = perfmetrics[str(perfmetric)]
    print('initial original testing data precision of cifar10_eval without alphastar on original training data',perf)
    finalresults.append((0, 0, 1, perf, perfmetrics, gen))
    
    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TrainSplit/' 
    labels = listdir(InDir)
    labels.sort()


    creator.create("FitnessMax", base.Fitness, weights=(0.0,),error=0.0)
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    toolbox.register("mutate", cifar10_adversary_train.mutation)
    toolbox.register("mate", cifar10_adversary_train.crossover)
    toolbox.register("evaluate", cifar10_adversary_train.evaluate)
    toolbox.register("select", cifar10_adversary_train.select)
    
    toolbox.register("individualImage", cifar10_adversary_train.initIndividualImage)
    toolbox.register("imagepopulation", cifar10_adversary_train.initImagePopulation, toolbox.individualImage)
    imagespopulation,positiveimagesmean = toolbox.imagepopulation(InDir)

    toolbox.register("attribute",cifar10_adversary_train.initIndividual, meanimage=positiveimagesmean)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=FLAGS.numalphas)
    alphaspopulation = toolbox.population()


#     print('imagespopulation',imagespopulation)
#     print('alphaspopulation',alphaspopulation)
#     print('alphaspopulation[0].fitness.weights',alphaspopulation[0].fitness.weights)

    print('Initialization completed')

#     sys.exit()
#     while(LoopingFlag and total_iters < maxiters):
    while(LoopingFlag and gen < FLAGS.numgens):
        print('gen',gen)
        
        
        cifar10_adversary_train.alphasfitnesses(alphaspopulation,imagespopulation,toolbox)
        print('len(alphaspopulation)',len(alphaspopulation))
        print('alphaspopulation selected for game',alphaspopulation)
        
        bestalphafitness = 0.0
        bestalpha = alphaspopulation[0]
        for index,_ in enumerate(alphaspopulation):
            print('alphaspopulation[index].fitness.weights selected for game',alphaspopulation[index].fitness.weights)
            if(alphaspopulation[index].fitness.weights > bestalphafitness):
                bestalphafitness = alphaspopulation[index].fitness.weights
                bestalpha = alphaspopulation[index]
        
        print('bestalpha selected for game',bestalpha)
        print('bestalphafitness selected for game',bestalphafitness)
        
        
#         total_iters = total_iters + 1
#         (alphaspopulation,imagespopulation) = cifar10_adversary_train.adversary_train_genetic(InDir,WeightsDir)
        
            
        curralpha = bestalpha
        adv_payoff = curralpha.fitness.weights[0]
        error = curralpha.fitness.error
        
        perfmetrics = {}
        perfmetrics['precision'] = curralpha.fitness.precision
        perfmetrics['recall'] = curralpha.fitness.recall
        perfmetrics['f1score'] = curralpha.fitness.f1score
        perfmetrics['tpr'] = curralpha.fitness.tpr
        perfmetrics['fpr'] = curralpha.fitness.fpr
        perf = perfmetrics[str(perfmetric)]
        
        print('curralpha',curralpha)
        print('curralpha.fitness.weights[0]',curralpha.fitness.weights[0])
        print('perf',perf)
        

        print('payoff: %f and performance: %f in iteration: %f' % (adv_payoff, perf, gen))
        finalresults.append((adv_payoff, error,1+error-adv_payoff, perf, perfmetrics, gen))
        print('finalresults',finalresults)

        if abs(adv_payoff - adv_payoff_highest) > FLAGS.myepsilon:
            adv_payoff_highest = adv_payoff

            selectedoffspring = toolbox.select(alphaspopulation)
            parents = cifar10_adversary_train.copyindividuals(selectedoffspring,toolbox)
            offspring = cifar10_adversary_train.copyindividuals(selectedoffspring,toolbox)

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                print('Calling mate')
                (child1, child2) = toolbox.mate(child1, child2)
                del child1.fitness.values
                child1.fitness.weights = (0.0,)
                del child2.fitness.values
                child2.fitness.weights = (0.0,)
                print('Reset mate weights')
                
                sys.exit()

            for mutant in offspring:
                print('Calling mutate')
                mutant = toolbox.mutate(mutant)
                del mutant.fitness.values
                mutant.fitness.weights = (0.0,)
                print('Reset mutant weights')
                
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if(len(invalid_ind) != 0):
                cifar10_adversary_train.alphasfitnesses(invalid_ind,imagespopulation,toolbox)
            alphaspopulation[:] = cifar10_adversary_train.copyindividuals(parents + offspring,toolbox)

            print('alphaspopulation',alphaspopulation)
            binarizer(GameInDir,AdvInDir,imagespopulation,curralpha,labels,'train.bin')
            if tf.gfile.Exists(TrainWeightsDir):
              tf.gfile.DeleteRecursively(TrainWeightsDir)
            tf.gfile.MakeDirs(TrainWeightsDir)
            cifar10_train.train()
            
        else:
            LoopingFlag = False
        
        gen = gen + 1 
    


    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TrainSplit/'
    imagespopulation,positiveimagesmean = toolbox.imagepopulation(InDir)
    binarizer(GameInDir,AdvInDir,imagespopulation,curralpha,labels,'train.bin')
    if tf.gfile.Exists(TrainWeightsDir):
      tf.gfile.DeleteRecursively(TrainWeightsDir)
    tf.gfile.MakeDirs(TrainWeightsDir)
    cifar10_train.train()
    
    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TestSplit/'
    imagespopulation,positiveimagesmean = toolbox.imagepopulation(InDir)
    binarizer(GameInDir,AdvInDir,imagespopulation,curralpha,labels,'test.bin')
    if tf.gfile.Exists(EvalDir):
      tf.gfile.DeleteRecursively(EvalDir)
    tf.gfile.MakeDirs(EvalDir)
    perfmetrics = cifar10_eval.evaluate()
    perf = perfmetrics[str(perfmetric)]
    error = curralpha.fitness.error
#     precision = cifar10_eval.evaluate()
#     distortedimages = []
#     for x in imagespopulation:
#         distortedimages.append((cifar10_adversary_train.distorted_image(x[1],bestalpha),x[0]))
#     precision = 1-cifar10_adversary_train.evaluate(distortedimages)
    print('final manipulated testing data precision of cifar10_eval with alphastar on manipulated training data',perf)
    finalresults.append((0, error, 1+error, perf, perfmetrics, gen))

    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/' 
    createdataset.binarizer(InDir,'TrainSplit/','train.bin')
    copyfile(InDir + 'train.bin', GameInDir + 'train.bin')
    if tf.gfile.Exists(TrainWeightsDir):
      tf.gfile.DeleteRecursively(TrainWeightsDir)
    tf.gfile.MakeDirs(TrainWeightsDir)
    cifar10_train.train()

    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TestSplit/'
    imagespopulation,positiveimagesmean = toolbox.imagepopulation(InDir)
    binarizer(GameInDir,AdvInDir,imagespopulation,curralpha,labels,'test.bin')
    if tf.gfile.Exists(EvalDir):
      tf.gfile.DeleteRecursively(EvalDir)
    tf.gfile.MakeDirs(EvalDir)
    perfmetrics = cifar10_eval.evaluate()
    perf = perfmetrics[str(perfmetric)]
    error = curralpha.fitness.error
#     precision = cifar10_eval.evaluate()
#     precision = 1-cifar10_adversary_train.evaluate(distortedimages)
    print('final manipulated testing data precision of cifar10_eval without alphastar on original training data',perf)
    finalresults.append((0, error, 1+error, perf, perfmetrics, gen))


    print('bestalpha',curralpha)
    print('adv_payoff_highest',adv_payoff_highest)
    print('gen',gen)
    print('FLAGS.numgens',FLAGS.numgens)
    print('FLAGS.numalphas',FLAGS.numalphas)
    print('FLAGS.myepsilon',FLAGS.myepsilon)
    print('FLAGS.mylambda',FLAGS.mylambda)
    print('finalresults',finalresults)

    
# wstar are neural network weights stored in files on disk
# Need to check whether game is converging as expected
    
#     curralpha = alphastar
#     binarizer(GameInDir,AdvInDir,imagespopulation,curralpha,labels,'test.bin')
#     if tf.gfile.Exists(EvalDir):
#       tf.gfile.DeleteRecursively(EvalDir)
#     tf.gfile.MakeDirs(EvalDir)
#     precision = cifar10_eval.evaluate()
#     print('final precision of cifar10_eval on alphastar',precision)

if __name__ == '__main__':
  tf.app.run()

