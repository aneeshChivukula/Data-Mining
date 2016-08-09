import tensorflow as tf
from root.cifar10 import cifar10

from root.cifar10 import cifar10_train
from root.cifar10 import cifar10_eval
from root.cifar10 import cifar10_adversary_train

def main(argv=None):  
    TrainDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_train'
    if tf.gfile.Exists(TrainDir):
      tf.gfile.DeleteRecursively(TrainDir)
    tf.gfile.MakeDirs(TrainDir)
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
    
    distortedimages = []
    for x in imagespopulation:
        distortedimages.append(cifar10_adversary_train.distorted_image(x[1],curralpha))
    print('final distortedimages',distortedimages)
    
    
    
    
    

if __name__ == '__main__':
  tf.app.run()

