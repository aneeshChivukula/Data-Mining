import tensorflow as tf
import numpy as np
import Image

from root.cifar10 import cifar10

from root.cifar10 import cifar10_train
from root.cifar10 import cifar10_eval
from root.cifar10 import cifar10_adversary_train
from root.cnns import createdataset

from os import listdir


def main(argv=None):  
    TrainWeightsDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_train'
    if tf.gfile.Exists(TrainWeightsDir):
      tf.gfile.DeleteRecursively(TrainWeightsDir)
    tf.gfile.MakeDirs(TrainWeightsDir)
    avg_loss = cifar10_train.train()

    EvalDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_eval'
    if tf.gfile.Exists(EvalDir):
      tf.gfile.DeleteRecursively(EvalDir)
    tf.gfile.MakeDirs(EvalDir)
    cifar10_eval.evaluate()
    
    maxiter = 10
    LoopingFlag = True
    total_iters = 0
    adv_payoff_highest = 0
    
    alphastar = np.zeros(size=(32, 32, 3))
    
    while(LoopingFlag and total_iters < maxiter):
        InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TrainSplit/' 
        WeightsDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_output'
        (alphaspopulation,imagespopulation) = cifar10_adversary_train.adversary_train_genetic(InDir,WeightsDir)
    
#         print('final alphaspopulation',alphaspopulation)
        curralpha = alphaspopulation[0]
        
        ls = listdir(InDir)
        ls.sort()
        
        AdvInDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/AdversarialSplit/'
        if tf.gfile.Exists(AdvInDir):
          tf.gfile.DeleteRecursively(AdvInDir)
        tf.gfile.MakeDirs(AdvInDir)
        
        for Label in ls:
            tf.gfile.MakeDirs(AdvInDir + Label)
    
        GameInDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_data/imagenet2010-batches-bin/'
        binfile = open(GameInDir + 'train.bin', 'wb',)
        L = []
    #     distortedimages = []
        for i,x in enumerate(imagespopulation):
            CurrLabel = ls[x[0]]
            CurrImage = np.array(cifar10_adversary_train.distorted_image(x[1],curralpha), np.uint8)[0]
            Image.fromarray(CurrImage).save(AdvInDir + CurrLabel + "/" + str(i) + ".jpeg")
            
            l = np.insert(CurrImage.flatten(order='F'),0, x[0])
            if(len(l) == createdataset.length):
                L.append(l)
    #         distortedimages.append(CurrImage)
    #     print('final distortedimages',distortedimages)
        
        np.concatenate(L).astype('int16').tofile(binfile)
        binfile.close()
        
        TrainWeightsDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_train'
        if tf.gfile.Exists(TrainWeightsDir):
          tf.gfile.DeleteRecursively(TrainWeightsDir)
        tf.gfile.MakeDirs(TrainWeightsDir)
        adv_payoff = cifar10_train.train()

        if adv_payoff > adv_payoff_highest:
            adv_payoff_highest = adv_payoff
            alphastar = alphastar + curralpha

            EvalDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_eval'
            if tf.gfile.Exists(EvalDir):
              tf.gfile.DeleteRecursively(EvalDir)
            tf.gfile.MakeDirs(EvalDir)
            cifar10_eval.evaluate()
        
        else:
            LoopingFlag = False
        
        total_iters = total_iters + 1
    
    print('alphastar',alphastar)
    # wstar are neural network weights stored in files on disk
    # Need to check whether game is converging as expected
    
    

if __name__ == '__main__':
  tf.app.run()

