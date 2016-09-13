import sys
import tensorflow as tf
import numpy as np
import PIL
from PIL import Image
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
import cPickle as pickle

FLAGS = tf.app.flags.FLAGS

# perfmetric = "precision"
perfmetric = "recall"
# perfmetric = "f1score"
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
    StdoutFile = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_train/Stdout.txt'
    AlphasFile = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_train/alphas.pkl'
    fp1 = open(StdoutFile,'wb')
    fp2 = open(AlphasFile,'wb')
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
    # For 2 class problem, change input bin file to include 2 classes and train over 2000 iterations
    # For 1000 class problem, change input bin file to include 1000 classes and train over 10000 iterations

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

#     createdataset.binarizer(InDir,'TrainSplit/','train.bin')
#     copyfile(InDir + 'train.bin', GameInDir + 'train.bin')
#     if tf.gfile.Exists(TrainWeightsDir):
#       tf.gfile.DeleteRecursively(TrainWeightsDir)
#     tf.gfile.MakeDirs(TrainWeightsDir)
#     cifar10_train.train()
    # IS this retraining needed?

#     createdataset.binarizer(InDir,'TestSplit/','test.bin')
#     copyfile(InDir + 'test.bin', GameInDir + 'test.bin')

    
    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TrainSplit/' 
    labels = listdir(InDir)
    labels.sort()


    creator.create("FitnessMax", base.Fitness, weights=(0.0,),error=0.0,precision=0.0,recall=0.0,f1score=0.0,tpr=0.0,fpr=0.0)
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
    pickle.dump(alphaspopulation,fp2)

#     print('imagespopulation',imagespopulation)
#     print('alphaspopulation',alphaspopulation)

    cifar10_adversary_train.alphasfitnesses(alphaspopulation,imagespopulation,toolbox)
    print('finalresults',finalresults)
#     sys.exit()
#     while(LoopingFlag and total_iters < maxiters):
    while(LoopingFlag and gen < FLAGS.numgens):
        print('gen',gen)
        
        bestalphafitness = 0.0
        bestalpha = alphaspopulation[0]
        for index,_ in enumerate(alphaspopulation):
            if(alphaspopulation[index].fitness.weights > bestalphafitness):
                bestalphafitness = alphaspopulation[index].fitness.weights
                bestalpha = alphaspopulation[index]
        
        print('bestalpha selected for game',bestalpha)
        print('bestalphafitness selected for game',bestalphafitness)
        
#         total_iters = total_iters + 1
#         (alphaspopulation,imagespopulation) = cifar10_adversary_train.adversary_train_genetic(InDir,WeightsDir)
            
        
        adv_payoff = bestalpha.fitness.weights[0]
        error = bestalpha.fitness.error
        
        perfmetrics = {}
        perfmetrics['precision'] = bestalpha.fitness.precision
        perfmetrics['recall'] = bestalpha.fitness.recall
        perfmetrics['f1score'] = bestalpha.fitness.f1score
        perfmetrics['tpr'] = bestalpha.fitness.tpr
        perfmetrics['fpr'] = bestalpha.fitness.fpr
        perf = perfmetrics[str(perfmetric)]

        print('payoff: %f and performance: %f in iteration: %f' % (adv_payoff, perf, gen))
        finalresults.append((adv_payoff, error,1+error-adv_payoff, perf, perfmetrics, gen))
        print('finalresults',finalresults)
        pickle.dump(finalresults,fp1)
        if abs(adv_payoff - adv_payoff_highest) > FLAGS.myepsilon:
            adv_payoff_highest = adv_payoff

            selectedoffspring = toolbox.select(alphaspopulation)
            parents = cifar10_adversary_train.copyindividuals(selectedoffspring,toolbox)
            offspring = cifar10_adversary_train.copyindividuals(selectedoffspring,toolbox)

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                print('Calling mate')

                (child1m,child2m) = toolbox.mate(child1, child2)
                child1[0] = np.copy(child1m[0])
                child2[0] = np.copy(child2m[0])
                child1 = toolbox.clone(child1)
                child2 = toolbox.clone(child2)
                
                
#                 (child1, child2) = tollbox.clone(toolbox.mate(child1, child2))
                del child1.fitness.values
                child1.fitness.weights = (0.0,)
                del child2.fitness.values
                child2.fitness.weights = (0.0,)
                print('Reset mate weights')
                print('child1.fitness.valid',child1.fitness.valid)
                print('child2.fitness.valid',child2.fitness.valid)
                
#                 sys.exit()

            for mutant in offspring:
                print('Calling mutate')

                mutantm = toolbox.mutate(mutant)
                mutant[0] = np.copy(mutantm[0])
                mutant = toolbox.clone(mutant)
                
#                 mutant = toolbox.mutate(mutant)
                del mutant.fitness.values
                mutant.fitness.weights = (0.0,)
                print('Reset mutant weights')
                print('mutant.fitness.valid',mutant.fitness.valid)

            binarizer(GameInDir,AdvInDir,imagespopulation,bestalpha,labels,'train.bin')
            if tf.gfile.Exists(TrainWeightsDir):
              tf.gfile.DeleteRecursively(TrainWeightsDir)
            tf.gfile.MakeDirs(TrainWeightsDir)
            cifar10_train.train()

            cifar10_adversary_train.alphasfitnesses(offspring,imagespopulation,toolbox)
#             invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#             if(len(invalid_ind) != 0):
#                 cifar10_adversary_train.alphasfitnesses(invalid_ind,imagespopulation,toolbox)
            alphaspopulation[:] = cifar10_adversary_train.copyindividuals(parents + offspring,toolbox)
            pickle.dump(alphaspopulation,fp2)
#             print('alphaspopulation',alphaspopulation)
            
            
            print('Iteration completed')
            
#             sys.exit()
            
        else:
            LoopingFlag = False
        
        gen = gen + 1 
    

    bestalphafitness = 0.0
    bestalpha = alphaspopulation[0]
    for index,_ in enumerate(alphaspopulation):
        if(alphaspopulation[index].fitness.weights > bestalphafitness):
            bestalphafitness = alphaspopulation[index].fitness.weights
            bestalpha = alphaspopulation[index]
    alphastar = bestalpha

    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TrainSplit/'
    imagespopulation,positiveimagesmean = toolbox.imagepopulation(InDir)
    binarizer(GameInDir,AdvInDir,imagespopulation,alphastar,labels,'train.bin')
    if tf.gfile.Exists(TrainWeightsDir):
      tf.gfile.DeleteRecursively(TrainWeightsDir)
    tf.gfile.MakeDirs(TrainWeightsDir)
    cifar10_train.train()
    
    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TestSplit/'
    imagespopulation,positiveimagesmean = toolbox.imagepopulation(InDir)
    binarizer(GameInDir,AdvInDir,imagespopulation,alphastar,labels,'test.bin')
    if tf.gfile.Exists(EvalDir):
      tf.gfile.DeleteRecursively(EvalDir)
    tf.gfile.MakeDirs(EvalDir)
    perfmetrics = cifar10_eval.evaluate()
    perf = perfmetrics[str(perfmetric)]
    error = alphastar.fitness.error
    adv_payoff = alphastar.fitness.weights[0]

#     precision = cifar10_eval.evaluate()
#     distortedimages = []
#     for x in imagespopulation:
#         distortedimages.append((cifar10_adversary_train.distorted_image(x[1],bestalpha),x[0]))
#     precision = 1-cifar10_adversary_train.evaluate(distortedimages)
    print('final manipulated testing data precision of cifar10_eval with alphastar on manipulated training data',perf)
    finalresults.append((adv_payoff, error, 1+error-adv_payoff, perf, perfmetrics, gen))

    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/' 
    createdataset.binarizer(InDir,'TrainSplit/','train.bin')
    copyfile(InDir + 'train.bin', GameInDir + 'train.bin')
    if tf.gfile.Exists(TrainWeightsDir):
      tf.gfile.DeleteRecursively(TrainWeightsDir)
    tf.gfile.MakeDirs(TrainWeightsDir)
    cifar10_train.train()

    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TestSplit/'
    imagespopulation,positiveimagesmean = toolbox.imagepopulation(InDir)
    binarizer(GameInDir,AdvInDir,imagespopulation,alphastar,labels,'test.bin')
    if tf.gfile.Exists(EvalDir):
      tf.gfile.DeleteRecursively(EvalDir)
    tf.gfile.MakeDirs(EvalDir)
    perfmetrics = cifar10_eval.evaluate()
    perf = perfmetrics[str(perfmetric)]
    error = alphastar.fitness.error
    adv_payoff = alphastar.fitness.weights[0]
#     precision = cifar10_eval.evaluate()
#     precision = 1-cifar10_adversary_train.evaluate(distortedimages)
    print('final manipulated testing data precision of cifar10_eval without alphastar on original training data',perf)
    finalresults.append((adv_payoff, error, 1+error-adv_payoff, perf, perfmetrics, gen))


    print('bestalpha',alphastar)
    print('bestalpha.fitness.weights[0]',alphastar.fitness.weights[0])
    print('adv_payoff_highest',adv_payoff_highest)
    print('gen',gen)
    print('FLAGS.numgens',FLAGS.numgens)
    print('FLAGS.numalphas',FLAGS.numalphas)
    print('FLAGS.myepsilon',FLAGS.myepsilon)
    print('FLAGS.mylambda',FLAGS.mylambda)
    print('finalresults',finalresults)
    
    pickle.dump(finalresults,fp1)

    fp1.close()
    fp2.close()
    
# wstar are neural network weights stored in files on disk
# Need to check whether game is converging as expected
    
#     alphastar = alphastar
#     binarizer(GameInDir,AdvInDir,imagespopulation,alphastar,labels,'test.bin')
#     if tf.gfile.Exists(EvalDir):
#       tf.gfile.DeleteRecursively(EvalDir)
#     tf.gfile.MakeDirs(EvalDir)
#     precision = cifar10_eval.evaluate()
#     print('final precision of cifar10_eval on alphastar',precision)

if __name__ == '__main__':
  tf.app.run()

