import tensorflow as tf
import numpy as np
import Image

from root.cifar10 import cifar10

from root.cifar10 import cifar10_train
from root.cifar10 import cifar10_eval
from root.cifar10 import cifar10_adversary_train
from root.cnns import createdataset

from os import listdir

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
    
    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/' 
    createdataset.binarizer(InDir,'TrainSplit/','train.bin')
    createdataset.binarizer(InDir,'TestSplit/','test.bin')
    
    TrainWeightsDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_train'
    if tf.gfile.Exists(TrainWeightsDir):
      tf.gfile.DeleteRecursively(TrainWeightsDir)
    tf.gfile.MakeDirs(TrainWeightsDir)
    avg_loss = cifar10_train.train()

    EvalDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_eval'
    if tf.gfile.Exists(EvalDir):
      tf.gfile.DeleteRecursively(EvalDir)
    tf.gfile.MakeDirs(EvalDir)
    precision = cifar10_eval.evaluate()
    
    eps = 0.0001
    maxiters = 10
    LoopingFlag = True
    total_iters = 0
    adv_payoff_highest = 0
    
    alphastar = np.zeros((32, 32, 3))
    finalresults = []
    
    finalresults.append((avg_loss, precision, total_iters))
    
    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TrainSplit/' 
    WeightsDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_output'
    AdvInDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/AdversarialSplit/'
    GameInDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_data/imagenet2010-batches-bin/'
    EvalDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_eval'
    TrainWeightsDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_train'
    labels = listdir(InDir)
    labels.sort()

    while(LoopingFlag and total_iters < maxiters):
        (alphaspopulation,imagespopulation) = cifar10_adversary_train.adversary_train_genetic(InDir,WeightsDir)
        curralpha = alphaspopulation[0]
    
        binarizer(GameInDir,AdvInDir,imagespopulation,curralpha,labels,'train.bin')
        
        if tf.gfile.Exists(TrainWeightsDir):
          tf.gfile.DeleteRecursively(TrainWeightsDir)
        tf.gfile.MakeDirs(TrainWeightsDir)
        adv_payoff = cifar10_train.train()

        if tf.gfile.Exists(EvalDir):
          tf.gfile.DeleteRecursively(EvalDir)
        tf.gfile.MakeDirs(EvalDir)
        precision = cifar10_eval.evaluate()

        print('payoff: %f and precision: %f in iteration: %f' % (adv_payoff, precision, total_iters))
        finalresults.append((adv_payoff, precision, total_iters))

        
        if abs(adv_payoff - adv_payoff_highest) > eps:
            adv_payoff_highest = adv_payoff
            alphastar = alphastar + curralpha
        else:
            LoopingFlag = False
        
        total_iters = total_iters + 1
    
    print('adv_payoff_highest',adv_payoff_highest)
    print('alphastar',alphastar)
    print('total_iters',total_iters)
    print('maxiters',maxiters)
    print('finalresults',finalresults)
    # wstar are neural network weights stored in files on disk
    # Need to check whether game is converging as expected
    
    curralpha = alphastar
    binarizer(GameInDir,AdvInDir,imagespopulation,curralpha,labels,'test.bin')

    if tf.gfile.Exists(EvalDir):
      tf.gfile.DeleteRecursively(EvalDir)
    tf.gfile.MakeDirs(EvalDir)
    precision = cifar10_eval.evaluate()
    print('final precision on alphastar',precision)




if __name__ == '__main__':
  tf.app.run()

