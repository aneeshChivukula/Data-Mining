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
    cifar10_train.train()
    
    EvalDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_eval'
    if tf.gfile.Exists(EvalDir):
      tf.gfile.DeleteRecursively(EvalDir)
    tf.gfile.MakeDirs(EvalDir)
    cifar10_eval.evaluate()

    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TrainSplit/' 
    WeightsDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_output'
    (alphaspopulation,imagespopulation) = cifar10_adversary_train.adversary_train_genetic(InDir,WeightsDir)

    print('final alphaspopulation',alphaspopulation)
    curralpha = alphaspopulation[0]
    
    ls = listdir(InDir)
    ls.sort()
    
    AdvInDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/AdversarialSplit/'
    if tf.gfile.Exists(AdvInDir):
      tf.gfile.DeleteRecursively(AdvInDir)
    tf.gfile.MakeDirs(AdvInDir)
    
    for Label in ls:
        tf.gfile.MakeDirs(AdvInDir + Label)
    
#     distortedimages = []
    for i,x in enumerate(imagespopulation):
        CurrLabel = ls[x[0]]
        CurrImage = np.array(cifar10_adversary_train.distorted_image(x[1],curralpha), np.uint8)[0]
        Image.fromarray(CurrImage).save(AdvInDir + CurrLabel + "/" + str(i) + ".jpeg")
#         distortedimages.append(CurrImage)
#     print('final distortedimages',distortedimages)
    
    
    
    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_data/'
    createdataset.binarizer(InDir,'imagenet2010-batches-bin/','train.bin')
    createdataset.binarizer(InDir,'imagenet2010-batches-bin/','test.bin')
    
    
    

if __name__ == '__main__':
  tf.app.run()

