from PIL import Image
import PIL
from deap import base
from deap import creator
from deap import tools
import math
from os import listdir
import os
import random
from shutil import copyfile
import shutil
import sys

import cPickle as pickle
import numpy as np
from root.cifar10 import cifar10
from root.cifar10 import cifar10_adversary_train
from root.cifar10 import cifar10_eval
from root.cifar10 import cifar10_train
from root.cnns import createdataset
import tensorflow as tf

import time
import heapq



FLAGS = tf.app.flags.FLAGS

# perfmetric = "precision"
perfmetric = "recall"
# perfmetric = "f1score"
# perfmetric = "tpr"
# perfmetric = "fpr"

# searchalg = "GA"
searchalg = "SA"

TempMax = 50
TempMin = 5
SampleSize = 5
ReductionRate = 0.1

executetwolabel = True
multiplayer = False # Make true only on server
trainmultiplayercnn = False

testgenadv = True
genimages=True

def load_gansamples(CurrDir):
    os.chdir(CurrDir)
    allsamples = []        
    for savedfile in sorted(os.listdir(CurrDir),key=lambda x: int(x.split('_')[-1].split('-')[0])):
        if savedfile.endswith(".pkl"):
            fp = open(savedfile,'rb')
            samples = pickle.load(fp)
            fp.close()
            allsamples.extend(samples)    
    return allsamples

def guidedmasking(mask):
    heightstartind = np.random.randint(low=0,high=32)
    heightendind = (heightstartind + np.random.randint(FLAGS.minheightlength,FLAGS.maxheightlength))
    while( heightendind > 32 ):
        heightstartind = np.random.randint(low=0,high=32)
        heightendind = (heightstartind + np.random.randint(FLAGS.minheightlength,FLAGS.maxheightlength))
    
    widthstartind = np.random.randint(low=0,high=32)
    widthendind = (widthstartind + np.random.randint(FLAGS.minwidthlength,FLAGS.maxwidthlength))
    while( widthendind > 32 ):
        widthstartind = np.random.randint(low=0,high=32)
        widthendind = (widthstartind + np.random.randint(FLAGS.minwidthlength,FLAGS.maxwidthlength))

    mask5 = np.zeros_like(mask)
    mask5[heightstartind:heightendind,widthstartind:widthendind,] = True
    mask = np.logical_and(mask,mask5)
    
    return mask
    


def transformer(AdvInDir,imagespopulation,curralpha,labels,filesd):

    if tf.gfile.Exists(AdvInDir):
      tf.gfile.DeleteRecursively(AdvInDir)
    tf.gfile.MakeDirs(AdvInDir)
    
    for Label in labels:
        tf.gfile.MakeDirs(AdvInDir + Label)
    
    for i,x in enumerate(imagespopulation):
        CurrLabel = labels[x[0]]
        if(x[0] == 0):
            CurrImage = np.array(cifar10_adversary_train.distorted_image(x[1],curralpha), np.uint8) # Use this line when CurrImage is saved to disk on postprocessing alphastar
            #CurrImage = np.array(cifar10_adversary_train.distorted_image(x[1],curralpha), np.uint8)[0]
        else:
            CurrImage = np.array(x[1], np.uint8)
#         Image.fromarray(CurrImage).save(AdvInDir + CurrLabel + "/" + str(i) + ".jpeg")


        CurrImage[CurrImage<0] = 0

        Image.fromarray(CurrImage).save(AdvInDir + CurrLabel + "/" + str(filesd[i]))

# def transformer(AdvInDir,imagespopulation,curralpha,labels,filesd):
#     if tf.gfile.Exists(AdvInDir):
#       tf.gfile.DeleteRecursively(AdvInDir)
#     tf.gfile.MakeDirs(AdvInDir)
#     
#     for Label in labels:
#         tf.gfile.MakeDirs(AdvInDir + Label)
#     
#     for i,x in enumerate(imagespopulation):
#         CurrLabel = labels[x[0]]
#         if(x[0] == 0):
#             CurrImage = np.array(cifar10_adversary_train.distorted_image(x[1],curralpha), np.uint8)[0]
#         else:
#             CurrImage = np.array(x[1], np.uint8)
# #         Image.fromarray(CurrImage).save(AdvInDir + CurrLabel + "/" + str(i) + ".jpeg")
#         Image.fromarray(CurrImage).save(AdvInDir + CurrLabel + "/" + str(filesd[i]) + ".jpeg")
#         print('AdvInDir + CurrLabel + "/" + str(filesd[i])',AdvInDir + CurrLabel + "/" + str(filesd[i]))

# def alphasaver(AdvInDir,curralpha,TempCurrent,idx):
#     CurrImage = np.array(curralpha, np.uint8)
#     
#     print('CurrImage',CurrImage)
#     Image.fromarray(CurrImage).save(AdvInDir + "/" + str(TempCurrent) + ":" + str(idx) + ".jpeg")

# def alphasaver(AdvInDir,curralpha,TempCurrent,idx):
#     fp = open(AdvInDir + "/" + str(TempCurrent) + ":" + str(idx) + ".pkl",'wb')
#     pickle.dump(np.array(curralpha, np.int32),fp)
# # curralpha = pickle.load(open('/scratch/cifar10_20/AdversarialSplitAlphan/0:0.pkl','rb'))
    

def alphasaver(AdvInDir,curralpha,TempCurrent,idx):
    CurrImage = np.array(curralpha, np.uint8)
    print('CurrImage',CurrImage)
    print('Image.fromarray(CurrImage)',Image.fromarray(CurrImage))
    Image.fromarray(CurrImage).save(AdvInDir + "/" + str(TempCurrent) + ":" + str(idx) + ".jpeg")
    

# def binarizermulti(GameInDir,AdvInDir,imagespopulation,curralphas,labels,infile):
#     binfile = open(GameInDir + infile, 'wb',)
#     L = []
# 
#     for curralpha in curralphas:
#         for i,x in enumerate(imagespopulation):
#             CurrLabel = labels[x[0]]
#             if(int(x[0]) == 0):
#                 CurrImage = np.array(cifar10_adversary_train.distorted_image(x[1],curralpha), np.uint8)[0]
#             else:
#                 CurrImage = np.array(x[1], np.uint8)
# 
#             l = np.insert(CurrImage.flatten(order='F'),0, x[0])
#             if(len(l) == createdataset.length):
#                 L.append(l)
# 
#     np.concatenate(L).astype('int16').tofile(binfile)
#     binfile.close()

def binarizermulti(GameInDir,AdvInDir,imagespopulation,curralphas,labels,infile,includetest):
    binfile = open(GameInDir + infile, 'wb',)
    L = []

    for curralpha in curralphas:
        for i,x in enumerate(imagespopulation):
            CurrLabel = labels[x[0]]
            if(int(x[0]) == 0):
                CurrImage = np.array(cifar10_adversary_train.distorted_image(x[1],curralpha), np.uint8)
            else:
                CurrImage = np.array(x[1], np.uint8)

            l = np.insert(CurrImage.flatten(order='F'),0, x[0])
            if(len(l) == createdataset.length):
                L.append(l)

    if(includetest==True):
        for i,x in enumerate(imagespopulation):
            CurrImage = np.array(x[1], np.uint8)
            l = np.insert(CurrImage.flatten(order='F'),0, x[0])
            if(len(l) == createdataset.length):
                L.append(l)



    np.concatenate(L).astype('int16').tofile(binfile)
    binfile.close()
    
def binarizerganmulti(GameInDir,AdvInDir,imagespopulation,curralphas,labels,infile,includetest,ganimages,ganlabels):
    binfile = open(GameInDir + infile, 'wb',)
    L = []

    for i,x in enumerate(imagespopulation):
        CurrLabel = labels[x[0]]
        CurrImage = np.array(x[1], np.uint8)
        
        l = np.insert(CurrImage.flatten(order='F'),0, x[0])
        if(len(l) == createdataset.length):
            L.append(l)
    
    ld = dict()
    sortganlabels = sorted(set(ganlabels),key=int)
    ld[sortganlabels[0]] = 0
    ld[sortganlabels[1]] = 1

    print('ld',ld)
    sys.exit()
    
    for image,label in zip(ganimages,ganlabels):
        CurrImage = np.array(image, np.uint8)
        CurrLabel = label


        l = np.insert(CurrImage.flatten(order='F'),0, ld[CurrLabel])
        if(len(l) == createdataset.length):
            L.append(l)

    np.concatenate(L).astype('int16').tofile(binfile)
    binfile.close()


    
#     for curralpha in curralphas:
#         for i,x in enumerate(imagespopulation):
#             CurrLabel = labels[x[0]]
#             if(int(x[0]) == 0):
#                 CurrImage = np.array(cifar10_adversary_train.distorted_image(x[1],curralpha), np.uint8)
#             else:
#                 CurrImage = np.array(x[1], np.uint8)
#                 
#             l = np.insert(CurrImage.flatten(order='F'),0, x[0])
#             if(len(l) == createdataset.length):
#                 L.append(l)
#         
#         ld = dict() # This code hack works only for two label problems. Need to make a longer ld reflecting sorted order for multilabel problems.
#         ld[sorted(set(ganlabels),key=int)[0]] = 0
#         ld[sorted(set(ganlabels),key=int)[1]] = 1
#         
#         for image,label in zip(ganimages,ganlabels):
#             print image.shape # Check imahe shape is suitable for cifar10_adversary_train.distorted_image
#             print curralpha.shape
#             sys.exit()
#             
#             CurrLabel = label
#             if(ld[label] == 0):
#                 CurrImage = np.array(cifar10_adversary_train.distorted_image(image,curralpha), np.uint8)
#             else:
#                 CurrImage = np.array(image, np.uint8)
#             
#             l = np.insert(CurrImage.flatten(order='F'),0, x[0])
#             if(len(l) == createdataset.length):
#                 L.append(l)
#             
#     if(includetest==True):
#         for i,x in enumerate(imagespopulation):
#             CurrImage = np.array(x[1], np.uint8)
#             l = np.insert(CurrImage.flatten(order='F'),0, x[0])
#             if(len(l) == createdataset.length):
#                 L.append(l)
# 
# 
#     np.concatenate(L).astype('int16').tofile(binfile)
#     binfile.close()






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
        if(int(x[0]) == 0):
            CurrImage = np.array(cifar10_adversary_train.distorted_image(x[1],curralpha), np.uint8)[0]
        else:
            CurrImage = np.array(x[1], np.uint8)
#             Image.fromarray(np.array(x[1], np.uint8)).save(AdvInDir + CurrLabel + "/" + str(i+2) + ".jpeg")

#         print('CurrImage',CurrImage)
#         print('type(CurrImage)',type(CurrImage))        
#         print('curralpha',curralpha)
#         print('imagespopulation[0]',imagespopulation[0])
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
    CheckpointsDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_train'
    InitialCheckpointsDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_train_initial'
#     InitialCheckpointsDir = '/scratch/cifar10_22/cifar10_train_initial'



    if(genimages==True): # Set all file paths appropriately
        InDir = '/scratch/cifar10_25/TrainSplit/'
        labels = listdir(InDir)
        labels.sort()
        
        creator.create("FitnessMax", base.Fitness, weights=(0.0,),error=0.0,precision=0.0,recall=0.0,f1score=0.0,tpr=0.0,fpr=0.0)
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        toolbox.register("mutate", cifar10_adversary_train.mutation)
        toolbox.register("mate", cifar10_adversary_train.crossover)
        toolbox.register("evaluate", cifar10_adversary_train.evaluate)
        toolbox.register("select", cifar10_adversary_train.select)
        toolbox.register("perturbate", cifar10_adversary_train.perturbation)
        
        toolbox.register("individualImage", cifar10_adversary_train.initIndividualImage)
        toolbox.register("imagepopulation", cifar10_adversary_train.initImagePopulation, toolbox.individualImage)
    
        toolbox.register("attribute",cifar10_adversary_train.initIndividual, meanimage=0)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=1)

        alphastar = pickle.load(open('/data/achivuku/Documents/MultiplayerGAAlphas/7and9/25/6:6.pkl','rb'))

        InDir = '/scratch/cifar10_25/TrainSplit/'
        imagespopulation,positiveimagesmean,negativeimagesmean,filesd = toolbox.imagepopulation(InDir)

        AdvInDirG = '/scratch/cifar10_25/AdversarialSplitAlphag/'

        transformer(AdvInDirG,imagespopulation,alphastar,labels,filesd)

        sys.exit()







    if(trainmultiplayercnn==True):
        if(os.path.exists(CheckpointsDir)):
            shutil.rmtree(CheckpointsDir)
            shutil.copytree(InitialCheckpointsDir,CheckpointsDir, CheckpointsDir)
    else:
        createdataset.binarizer(InDir,'TrainSplit/','train.bin')
        copyfile(InDir + 'train.bin', GameInDir + 'train.bin')
        if tf.gfile.Exists(TrainWeightsDir):
          tf.gfile.DeleteRecursively(TrainWeightsDir)
        tf.gfile.MakeDirs(TrainWeightsDir)
        cifar10_train.train()
        if(os.path.exists(InitialCheckpointsDir)):
            shutil.rmtree(InitialCheckpointsDir)
        shutil.copytree(CheckpointsDir, InitialCheckpointsDir)
        # For 2 class problem, change input bin file to include 2 classes and train over 2000 iterations
        # For 1000 class problem, change input bin file to include 1000 classes and train over 10000 iterations


#     if(trainmultiplayercnn==True):
#         InitialCheckpointsDir = '/scratch/cifar10_22/cifar10_train_initial'
#         if(os.path.exists(CheckpointsDir)):
#             shutil.rmtree(CheckpointsDir)
#             shutil.copytree(InitialCheckpointsDir,CheckpointsDir, CheckpointsDir)
#     else:
#         InDir = '/scratch/cifar10_23/' 
#         createdataset.binarizer(InDir,'TrainSplit/','train.bin')
#         copyfile(InDir + 'train.bin', GameInDir + 'train.bin')
#         if tf.gfile.Exists(TrainWeightsDir):
#           tf.gfile.DeleteRecursively(TrainWeightsDir)
#         tf.gfile.MakeDirs(TrainWeightsDir)
#         cifar10_train.train()
#         if(os.path.exists(InitialCheckpointsDir)):
#             shutil.rmtree(InitialCheckpointsDir)
#         shutil.copytree(CheckpointsDir, InitialCheckpointsDir)
#         # For 2 class problem, change input bin file to include 2 classes and train over 2000 iterations
#         # For 1000 class problem, change input bin file to include 1000 classes and train over 10000 iterations

    if(testgenadv==True): # Set all file paths appropriately
        InDir = '/scratch/cifar10_25/TrainSplit/'
        labels = listdir(InDir)
        labels.sort()
        
        creator.create("FitnessMax", base.Fitness, weights=(0.0,),error=0.0,precision=0.0,recall=0.0,f1score=0.0,tpr=0.0,fpr=0.0)
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        toolbox.register("mutate", cifar10_adversary_train.mutation)
        toolbox.register("mate", cifar10_adversary_train.crossover)
        toolbox.register("evaluate", cifar10_adversary_train.evaluate)
        toolbox.register("select", cifar10_adversary_train.select)
        toolbox.register("perturbate", cifar10_adversary_train.perturbation)
        
        toolbox.register("individualImage", cifar10_adversary_train.initIndividualImage)
        toolbox.register("imagepopulation", cifar10_adversary_train.initImagePopulation, toolbox.individualImage)
    
        toolbox.register("attribute",cifar10_adversary_train.initIndividual, meanimage=0)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=1)


        '''

        alphastar1 = pickle.load(open('/scratch/cifar10_26/AdversarialSplitAlphac/7:7.pkl','rb'))
        alphastar2 = pickle.load(open('/scratch/cifar10_25/AdversarialSplitAlphac/6:6.pkl','rb'))
        alphastar3 = pickle.load(open('/scratch/cifar10_24/AdversarialSplitAlphac/4:4.pkl','rb'))
        alphastar4 = pickle.load(open('/scratch/cifar10_23/AdversarialSplitAlphac/4:4.pkl','rb'))
        alphastar5 = pickle.load(open('/scratch/cifar10_22/AdversarialSplitAlphac/2:2.pkl','rb'))
        
        '''
        # TO DO : After checking code correctness, generate data for (4,9) not (7,9)
        alphastars = []
        alphastar1 = pickle.load(open('/scratch/cifar10_26/AdversarialSplitAlphac/3:3.pkl','rb'))
        alphastar2 = pickle.load(open('/scratch/cifar10_25/AdversarialSplitAlphac/9:9.pkl','rb'))
        alphastar3 = pickle.load(open('/scratch/cifar10_24/AdversarialSplitAlphac/3:3.pkl','rb'))
        alphastar4 = pickle.load(open('/scratch/cifar10_23/AdversarialSplitAlphac/4:4.pkl','rb'))
        alphastar5 = pickle.load(open('/scratch/cifar10_22/AdversarialSplitAlphac/7:7.pkl','rb'))

        alphastars.append(alphastar1)
        alphastars.append(alphastar2)
        alphastars.append(alphastar3)
        alphastars.append(alphastar4)
        alphastars.append(alphastar5)
        
        finalresultsmulti = []

        FLAGS.max_iter_eval = 10000
        
        CurrDir = '/home/achivuku/Desktop/Conditional-DCGAN-master/samples/images'
        ganimages = load_gansamples(CurrDir)
        CurrDir = '/home/achivuku/Desktop/Conditional-DCGAN-master/samples/labels'
        ganlabels = load_gansamples(CurrDir)

        includetest = False
        InDir = '/scratch/cifar10_25/TrainSplit/'
        imagespopulation,positiveimagesmean,negativeimagesmean,filesd = toolbox.imagepopulation(InDir)
        binarizerganmulti(GameInDir,AdvInDir,imagespopulation,alphastars,labels,'train.bin',includetest,ganimages,ganlabels)        
        if tf.gfile.Exists(TrainWeightsDir):
            tf.gfile.DeleteRecursively(TrainWeightsDir)
        tf.gfile.MakeDirs(TrainWeightsDir)
        cifar10_train.train()

        includetest = True
        InDir = '/scratch/cifar10_25/TestSplit/'
        imagespopulation,positiveimagesmean,negativeimagesmean,filesd = toolbox.imagepopulation(InDir)
        binarizermulti(GameInDir,AdvInDir,imagespopulation,alphastars,labels,'test.bin',includetest)
        if tf.gfile.Exists(EvalDir):
          tf.gfile.DeleteRecursively(EvalDir)
        tf.gfile.MakeDirs(EvalDir)
        perfmetrics = cifar10_eval.evaluate()

        finalresultsmulti.append(perfmetrics)
        
        print('finalresultsmulti',finalresultsmulti)
        sys.exit()

    if(multiplayer==True): # Set all file paths appropriately
        InDir = '/scratch/cifar10_25/TrainSplit/'
        labels = listdir(InDir)
        labels.sort()


        creator.create("FitnessMax", base.Fitness, weights=(0.0,),error=0.0,precision=0.0,recall=0.0,f1score=0.0,tpr=0.0,fpr=0.0)
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        toolbox.register("mutate", cifar10_adversary_train.mutation)
        toolbox.register("mate", cifar10_adversary_train.crossover)
        toolbox.register("evaluate", cifar10_adversary_train.evaluate)
        toolbox.register("select", cifar10_adversary_train.select)
        toolbox.register("perturbate", cifar10_adversary_train.perturbation)
        
        toolbox.register("individualImage", cifar10_adversary_train.initIndividualImage)
        toolbox.register("imagepopulation", cifar10_adversary_train.initImagePopulation, toolbox.individualImage)
    
        toolbox.register("attribute",cifar10_adversary_train.initIndividual, meanimage=0)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=1)


        alphastars = []



        '''

        alphastar1 = pickle.load(open('/scratch/cifar10_26/AdversarialSplitAlphac/7:7.pkl','rb'))
        alphastar2 = pickle.load(open('/scratch/cifar10_25/AdversarialSplitAlphac/6:6.pkl','rb'))
        alphastar3 = pickle.load(open('/scratch/cifar10_24/AdversarialSplitAlphac/4:4.pkl','rb'))
        alphastar4 = pickle.load(open('/scratch/cifar10_23/AdversarialSplitAlphac/4:4.pkl','rb'))
        alphastar5 = pickle.load(open('/scratch/cifar10_22/AdversarialSplitAlphac/2:2.pkl','rb'))
        
        '''


        alphastar1 = pickle.load(open('/scratch/cifar10_31/AdversarialSplitAlphac/7:7.pkl','rb'))
#        alphastar2 = pickle.load(open('/scratch/cifar10_30/AdversarialSplitAlphac/6:6.pkl','rb'))
#        alphastar3 = pickle.load(open('/scratch/cifar10_29/AdversarialSplitAlphac/4:4.pkl','rb'))
        alphastar4 = pickle.load(open('/scratch/cifar10_28/AdversarialSplitAlphac/4:4.pkl','rb'))
#        alphastar5 = pickle.load(open('/scratch/cifar10_27/AdversarialSplitAlphac/2:2.pkl','rb'))

        finalresultsmulti = []


        '''
        InDir = '/scratch/cifar10_22/' 
        createdataset.binarizer(InDir,'TestSplit/','test.bin')
        copyfile(InDir + 'test.bin', GameInDir + 'test.bin')
        if tf.gfile.Exists(EvalDir):
          tf.gfile.DeleteRecursively(EvalDir)
        tf.gfile.MakeDirs(EvalDir)
        perfmetrics = cifar10_eval.evaluate()
        finalresultsmulti.append(perfmetrics)

        InDir = '/scratch/cifar10_22/TestSplit/'
        imagespopulation,positiveimagesmean,negativeimagesmean,filesd = toolbox.imagepopulation(InDir)
        binarizer(GameInDir,AdvInDir,imagespopulation,alphastar5,labels,'test.bin')
        if tf.gfile.Exists(EvalDir):
          tf.gfile.DeleteRecursively(EvalDir)
        tf.gfile.MakeDirs(EvalDir)
        perfmetrics = cifar10_eval.evaluate()
        finalresultsmulti.append(perfmetrics)

        print(finalresultsmulti)

        sys.exit()
        
        '''

        alphastars.append(alphastar1)
#        alphastars.append(alphastar2)
#        alphastars.append(alphastar3)
        alphastars.append(alphastar4)
#        alphastars.append(alphastar5)
        
#        InDir = '/scratch/cifar10_26/' 
#        createdataset.binarizer(InDir,'TrainSplit/','train.bin')
#        copyfile(InDir + 'train.bin', GameInDir + 'train.bin')
#        if tf.gfile.Exists(TrainWeightsDir):
#          tf.gfile.DeleteRecursively(TrainWeightsDir)
#        tf.gfile.MakeDirs(TrainWeightsDir)
#        cifar10_train.train()
        if(os.path.exists(CheckpointsDir)):
            shutil.rmtree(CheckpointsDir)
            shutil.copytree(InitialCheckpointsDir,CheckpointsDir, CheckpointsDir)

        FLAGS.max_iter_eval = 10000
        
        includetest = False
        InDir = '/scratch/cifar10_25/TestSplit/'
        imagespopulation,positiveimagesmean,negativeimagesmean,filesd = toolbox.imagepopulation(InDir)
        binarizermulti(GameInDir,AdvInDir,imagespopulation,alphastars,labels,'test.bin',includetest)
        if tf.gfile.Exists(EvalDir):
          tf.gfile.DeleteRecursively(EvalDir)
        tf.gfile.MakeDirs(EvalDir)
        perfmetrics = cifar10_eval.evaluate()

        finalresultsmulti.append(perfmetrics)

        includetest = False
        InDir = '/scratch/cifar10_25/TrainSplit/'
        imagespopulation,positiveimagesmean,negativeimagesmean,filesd = toolbox.imagepopulation(InDir)
        binarizermulti(GameInDir,AdvInDir,imagespopulation,alphastars,labels,'train.bin',includetest)
        if tf.gfile.Exists(TrainWeightsDir):
            tf.gfile.DeleteRecursively(TrainWeightsDir)
        tf.gfile.MakeDirs(TrainWeightsDir)
        cifar10_train.train()

        includetest = True
        InDir = '/scratch/cifar10_25/TestSplit/'
        imagespopulation,positiveimagesmean,negativeimagesmean,filesd = toolbox.imagepopulation(InDir)
        binarizermulti(GameInDir,AdvInDir,imagespopulation,alphastars,labels,'test.bin',includetest)
        if tf.gfile.Exists(EvalDir):
          tf.gfile.DeleteRecursively(EvalDir)
        tf.gfile.MakeDirs(EvalDir)
        perfmetrics = cifar10_eval.evaluate()

        finalresultsmulti.append(perfmetrics)

        print('finalresultsmulti',finalresultsmulti)
        sys.exit()

    
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

    StdoutFile = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_train/Stdout.txt'
    AlphasFile = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_train/alphas.pkl'
    FinalAlphaFile = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_train/alphac.pkl'    
    fp1 = open(StdoutFile,'wb')
    fp2 = open(AlphasFile,'wb')
    fp3 = open(FinalAlphaFile,'wb')    
 
# Comment while testing on laptop
 
#     InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/' 
#     InDir = '/scratch/cifar10_22/' 
#     createdataset.binarizer(InDir,'TrainSplit/','test.bin')
#     copyfile(InDir + 'test.bin', GameInDir + 'test.bin')
#     if tf.gfile.Exists(EvalDir):
#       tf.gfile.DeleteRecursively(EvalDir)
#     tf.gfile.MakeDirs(EvalDir)
#     perfmetrics = cifar10_eval.evaluate()
#     perf = perfmetrics[str(perfmetric)]
#     print('initial original training data performance of cifar10_eval without alphastar on original training data',perf)
#     finalresults.append((1, 0, 1, perf, perfmetrics, gen))
#      
#        
#     createdataset.binarizer(InDir,'TestSplit/','test.bin')
#     copyfile(InDir + 'test.bin', GameInDir + 'test.bin')
#     if tf.gfile.Exists(EvalDir):
#       tf.gfile.DeleteRecursively(EvalDir)
#     tf.gfile.MakeDirs(EvalDir)
#     perfmetrics = cifar10_eval.evaluate()
#     perf = perfmetrics[str(perfmetric)]
#     print('initial original testing data precision of cifar10_eval without alphastar on original training data',perf)
#     finalresults.append((0, 0, 1, perf, perfmetrics, gen))

# Comment while testing on laptop
 
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
    toolbox.register("perturbate", cifar10_adversary_train.perturbation)
    
    toolbox.register("individualImage", cifar10_adversary_train.initIndividualImage)
    toolbox.register("imagepopulation", cifar10_adversary_train.initImagePopulation, toolbox.individualImage)
#     imagespopulation,positiveimagesmean,negativeimagesmean,imagesbyclass,filesd = toolbox.imagepopulation(InDir)
    imagespopulation,positiveimagesmean,negativeimagesmean,filesd = toolbox.imagepopulation(InDir)
    
    

#     toolbox.register("attribute",cifar10_adversary_train.initIndividual, meanimage=positiveimagesmean)
    toolbox.register("attribute",cifar10_adversary_train.initIndividual, meanimage=0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=1)



    if(searchalg=="SA"):
        mask1 = positiveimagesmean != 0
        mask2 = negativeimagesmean != 0

        print('len(positiveimagesmean[positiveimagesmean<0])',len(positiveimagesmean[positiveimagesmean<0]))
        print('len(positiveimagesmean[positiveimagesmean!=0])',len(positiveimagesmean[positiveimagesmean!=0]))
        print('len(positiveimagesmean[positiveimagesmean<=math.ceil(FLAGS.high/FLAGS.dividend)] & positiveimagesmean[positiveimagesmean>0])',len(positiveimagesmean[np.logical_and(positiveimagesmean<=math.ceil(FLAGS.high/FLAGS.dividend),positiveimagesmean>0)]))

        positiveval = min(heapq.nlargest(FLAGS.positiveintensitysize, np.ndarray.flatten(positiveimagesmean[:,:,0])))
        mask3 = positiveimagesmean >= positiveval
        negativeval = min(heapq.nlargest(FLAGS.negativeintensitysize, np.ndarray.flatten(negativeimagesmean[:,:,0])))
        mask4 = negativeimagesmean >= negativeval

#         freqmask = np.zeros((32, 32, 3))
#         
#         for i in xrange(0,10):
#             freqmask = freqmask + (imagesbyclass[i] != 0).astype(int)
#         print('freqmask',freqmask)
#         mask5 = freqmask > FLAGS.minclassfreq
            
#         mask = mask2
#         mask = np.logical_and(mask1,mask2)
#         mask = np.logical_and.reduce((mask1,mask2,mask3,mask4))
#        mask = np.logical_and.reduce((mask1,mask2,mask3,mask4,mask5))
        masko = np.logical_or(mask3,mask4)
#         masko = np.logical_xor(mask3,mask4)
        
        mask = guidedmasking(masko)
#         while(np.count_nonzero(mask5) < (FLAGS.negativeintensitysize - FLAGS.positiveintensitysize)):
        while(np.count_nonzero(mask) < 100):
            mask = guidedmasking(masko)
        
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=1)
        alphac = toolbox.population() # 'alphac[0].shape', (1, 32, 32, 3))
        alphac[0][0][np.logical_not(mask)] = 0

        while(np.count_nonzero(alphac) == 0):
            alphac = toolbox.population() # 'alphac[0].shape', (1, 32, 32, 3))
            alphac[0][0][np.logical_not(mask)] = 0


#        pickle.dump(alphac,fp2)

#        binarizer(GameInDir,AdvInDir,imagespopulation,alphac[0],labels,'train.bin')

#        cifar10_adversary_train.alphafitness(alphac,imagespopulation,toolbox)
        evalc = alphac[0].fitness.weights[0]
        alphag = alphac
        alphan = alphac
        
        AdvInDirN = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/AdversarialSplitAlphan/'
        AdvInDirG = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/AdversarialSplitAlphag/'
        AdvInDirC = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/AdversarialSplitAlphac/'

        if tf.gfile.Exists(AdvInDirN):
            tf.gfile.DeleteRecursively(AdvInDirN)
        tf.gfile.MakeDirs(AdvInDirN)

        if tf.gfile.Exists(AdvInDirG):
            tf.gfile.DeleteRecursively(AdvInDirG)
        tf.gfile.MakeDirs(AdvInDirG)

        if tf.gfile.Exists(AdvInDirC):
            tf.gfile.DeleteRecursively(AdvInDirC)
        tf.gfile.MakeDirs(AdvInDirC)

        
        while(LoopingFlag):

            print('gen',gen)
        
            TempCurrent = TempMax
            
            evalg = alphag[0].fitness.weights[0]
            adv_payoff = round(evalg,FLAGS.numdecimalplaces)
            
            print('adv_payoff - adv_payoff_highest',adv_payoff - adv_payoff_highest)
            
#            if abs(adv_payoff - adv_payoff_highest) > FLAGS.myepsilon:
            if True:
                adv_payoff_highest = adv_payoff
    
                print('adv_payoff_highest',adv_payoff_highest)
                
                while TempCurrent >= TempMin:
                    for idx in xrange(0,SampleSize):
                        start_time = time.time()
                        print('idx',idx)
                        mutantm = toolbox.perturbate(alphac[0],mask)
                        alphan[0][0] = np.copy(mutantm[0])
                        alphan = toolbox.clone(alphan)
                        # Weights are NOT retained after cloning. Weights must be recomputed if cloning is used for deep copy
                        
                        
                        alphasaver(AdvInDirN,alphan[0][0],TempCurrent,idx)
                        sys.exit()
                        
                        cifar10_adversary_train.alphafitness(alphan,imagespopulation,toolbox)
                        
                        evaln = alphan[0].fitness.weights[0]
                        print('evaln',round(evaln,FLAGS.numdecimalplaces)) 
                        print('evalg',round(evalg,FLAGS.numdecimalplaces)) 
                        print('evalc',round(evalc,FLAGS.numdecimalplaces))
                        print('TempCurrent',TempCurrent)
                        
                        if evaln > evalc:
                            print('In first if')
                            alphac = alphan
                            evalc = evaln
                            if evalg < evaln:
                                print('In second if')
                                alphag = alphan
                                evalg = evaln
                        elif random.random() <= math.exp((evaln-evalc)/TempCurrent):
                            print('In third if')
                            alphac = alphan
                            evalc = evaln
                        print("--- %s seconds ---" % (time.time() - start_time))
                    print('TempCurrent',TempCurrent)
                    TempCurrent *= ReductionRate
                print('End Game Iteration')
    
                print('alphastar.fitness.weights',alphag[0].fitness.weights)

                adv_payoff = round(alphag[0].fitness.weights[0],FLAGS.numdecimalplaces)
                error = round(alphag[0].fitness.error,FLAGS.numdecimalplaces)

                perfmetrics = {}
                perfmetrics['precision'] = round(alphag[0].fitness.precision,FLAGS.numdecimalplaces)
                perfmetrics['recall'] = round(alphag[0].fitness.recall,FLAGS.numdecimalplaces)
                perfmetrics['f1score'] = round(alphag[0].fitness.f1score,FLAGS.numdecimalplaces)
                perfmetrics['tpr'] = round(alphag[0].fitness.tpr,FLAGS.numdecimalplaces)
                perfmetrics['fpr'] = round(alphag[0].fitness.fpr,FLAGS.numdecimalplaces)
                perf = round(perfmetrics[str(perfmetric)],FLAGS.numdecimalplaces)
    
                print('payoff: %f and performance: %f in iteration: %f' % (adv_payoff, perf, gen))

                if(executetwolabel == True):
                    finalresults.append((adv_payoff, error,round(1+error-adv_payoff,FLAGS.numdecimalplaces), perf, perfmetrics, gen))
                else:
                    finalresults.append((adv_payoff, error,round(error-adv_payoff,FLAGS.numdecimalplaces), perf, perfmetrics, gen))
                    
                print('finalresults',finalresults)
                pickle.dump(finalresults,fp1)
                

                binarizer(GameInDir,AdvInDir,imagespopulation,alphag[0],labels,'train.bin')
                if tf.gfile.Exists(TrainWeightsDir):
                  tf.gfile.DeleteRecursively(TrainWeightsDir)
                tf.gfile.MakeDirs(TrainWeightsDir)
                cifar10_train.train()
    
                alphac = alphag 
                gen = gen + 1 
                
                
                transformer(AdvInDirG,imagespopulation,alphag[0][0],labels,filesd)
                alphasaver(AdvInDirC,alphac[0][0],TempCurrent,idx)
                print('Iteration completed')
                sys.exit()
            else:
                LoopingFlag = False

        
        print('Game completed')
        alphastar = alphag[0]
        print('alphastar[0].fitness.weights[0]',alphastar.fitness.weights[0])
        print('alphastar[0].fitness.precision',alphastar.fitness.precision)
        print('alphastar[0].fitness.recall',alphastar.fitness.recall)
        print('alphastar[0].fitness.f1score',alphastar.fitness.f1score)
        print('alphastar[0].fitness.tpr',alphastar.fitness.tpr)
        print('alphastar[0].fitness.fpr',alphastar.fitness.fpr)
        pickle.dump(alphastar,fp3)
    
    elif(searchalg=="GA"):

        AdvInDirN = '/scratch/cifar10_26/AdversarialSplitAlphan/'
        AdvInDirG = '/scratch/cifar10_26/AdversarialSplitAlphag/'
        AdvInDirC = '/scratch/cifar10_26/AdversarialSplitAlphac/'

        if tf.gfile.Exists(AdvInDirN):
            tf.gfile.DeleteRecursively(AdvInDirN)
        tf.gfile.MakeDirs(AdvInDirN)

        if tf.gfile.Exists(AdvInDirG):
            tf.gfile.DeleteRecursively(AdvInDirG)
        tf.gfile.MakeDirs(AdvInDirG)

        if tf.gfile.Exists(AdvInDirC):
            tf.gfile.DeleteRecursively(AdvInDirC)
        tf.gfile.MakeDirs(AdvInDirC)


        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=FLAGS.numalphas)
        alphaspopulation = toolbox.population()
        pickle.dump(alphaspopulation,fp2)
        
        
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

            alphasaver(AdvInDirN,bestalpha[0],gen,gen)
            
            print('bestalpha selected for game',bestalpha)
            print('bestalphafitness selected for game',bestalphafitness)
            
    #         total_iters = total_iters + 1
    #         (alphaspopulation,imagespopulation) = cifar10_adversary_train.adversary_train_genetic(InDir,WeightsDir)
                
            
            adv_payoff = round(bestalpha.fitness.weights[0],FLAGS.numdecimalplaces)
            error = round(bestalpha.fitness.error,FLAGS.numdecimalplaces)
            
            perfmetrics = {}
            perfmetrics['precision'] = round(bestalpha.fitness.precision,FLAGS.numdecimalplaces)
            perfmetrics['recall'] = round(bestalpha.fitness.recall,FLAGS.numdecimalplaces)
            perfmetrics['f1score'] = round(bestalpha.fitness.f1score,FLAGS.numdecimalplaces)
            perfmetrics['tpr'] = round(bestalpha.fitness.tpr,FLAGS.numdecimalplaces)
            perfmetrics['fpr'] = round(bestalpha.fitness.fpr,FLAGS.numdecimalplaces)
            perf = round(perfmetrics[str(perfmetric)],FLAGS.numdecimalplaces)
    
            print('payoff: %f and performance: %f in iteration: %f' % (adv_payoff, perf, gen))
            if(executetwolabel == True):
                finalresults.append((adv_payoff, error,round(1+error-adv_payoff,FLAGS.numdecimalplaces), perf, perfmetrics, gen))
            else:
                finalresults.append((adv_payoff, error,round(error-adv_payoff,FLAGS.numdecimalplaces), perf, perfmetrics, gen))
            print('finalresults',finalresults)
            pickle.dump(finalresults,fp1)
            if abs(adv_payoff - adv_payoff_highest) > FLAGS.myepsilon:
                adv_payoff_highest = adv_payoff
    
                parents,selectedoffspring = toolbox.select(alphaspopulation)
    #             parents = cifar10_adversary_train.copyindividuals(selectedparents,toolbox)
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
    imagespopulation,positiveimagesmean,negativeimagesmean,imagesbyclass,filesd = toolbox.imagepopulation(InDir)
    binarizer(GameInDir,AdvInDir,imagespopulation,alphastar,labels,'train.bin')
    if tf.gfile.Exists(TrainWeightsDir):
      tf.gfile.DeleteRecursively(TrainWeightsDir)
    tf.gfile.MakeDirs(TrainWeightsDir)
    cifar10_train.train()
    
    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TestSplit/'
    imagespopulation,positiveimagesmean,negativeimagesmean,imagesbyclass,filesd = toolbox.imagepopulation(InDir)
    binarizer(GameInDir,AdvInDir,imagespopulation,alphastar,labels,'test.bin')
    if tf.gfile.Exists(EvalDir):
      tf.gfile.DeleteRecursively(EvalDir)
    tf.gfile.MakeDirs(EvalDir)
    perfmetrics = cifar10_eval.evaluate()
    perf = round(perfmetrics[str(perfmetric)],FLAGS.numdecimalplaces)
    error = round(alphastar.fitness.error,FLAGS.numdecimalplaces)
    adv_payoff = round(alphastar.fitness.weights[0],FLAGS.numdecimalplaces)

#     precision = cifar10_eval.evaluate()
#     distortedimages = []
#     for x in imagespopulation:
#         distortedimages.append((cifar10_adversary_train.distorted_image(x[1],bestalpha),x[0]))
#     precision = 1-cifar10_adversary_train.evaluate(distortedimages)
    print('final manipulated testing data precision of cifar10_eval with alphastar on manipulated training data',perf)

    if(executetwolabel == True):
        finalresults.append((adv_payoff, error, round(1+error-adv_payoff,FLAGS.numdecimalplaces), perf, perfmetrics, gen))
    else:
        finalresults.append((adv_payoff, error,round(error-adv_payoff,FLAGS.numdecimalplaces), perf, perfmetrics, gen))
        
#     InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/' 
#     createdataset.binarizer(InDir,'TrainSplit/','train.bin')
#     copyfile(InDir + 'train.bin', GameInDir + 'train.bin')
#     if tf.gfile.Exists(TrainWeightsDir):
#       tf.gfile.DeleteRecursively(TrainWeightsDir)
#     tf.gfile.MakeDirs(TrainWeightsDir)
#     cifar10_train.train()
    if(os.path.exists(CheckpointsDir)):
        shutil.rmtree(CheckpointsDir)
    shutil.copytree(InitialCheckpointsDir,CheckpointsDir, CheckpointsDir)

    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TestSplit/'
    imagespopulation,positiveimagesmean,negativeimagesmean,imagesbyclass,filesd = toolbox.imagepopulation(InDir)
    binarizer(GameInDir,AdvInDir,imagespopulation,alphastar,labels,'test.bin')
    if tf.gfile.Exists(EvalDir):
      tf.gfile.DeleteRecursively(EvalDir)
    tf.gfile.MakeDirs(EvalDir)
    perfmetrics = cifar10_eval.evaluate()
    perf = round(perfmetrics[str(perfmetric)],FLAGS.numdecimalplaces)
    error = round(alphastar.fitness.error,FLAGS.numdecimalplaces)
    adv_payoff = round(alphastar.fitness.weights[0],FLAGS.numdecimalplaces)
#     precision = cifar10_eval.evaluate()
#     precision = 1-cifar10_adversary_train.evaluate(distortedimages)

    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TrainSplit/'
    AdvInDirTrain = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/AdversarialSplitTrain/'
    imagespopulation,positiveimagesmean,negativeimagesmean,imagesbyclass,filesd = toolbox.imagepopulation(InDir)
    transformer(AdvInDirTrain,imagespopulation,alphastar[0][0],labels,filesd)

    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TestSplit/'
    AdvInDirTest = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/AdversarialSplitTest/'
    imagespopulation,positiveimagesmean,negativeimagesmean,imagesbyclass,filesd = toolbox.imagepopulation(InDir)
    transformer(AdvInDirTest,imagespopulation,alphastar[0][0],labels,filesd)

    print('final manipulated testing data precision of cifar10_eval without alphastar on original training data',perf)
    
    if(executetwolabel == True):
        finalresults.append((adv_payoff, error, round(1+error-adv_payoff,FLAGS.numdecimalplaces), perf, perfmetrics, gen))
    else:
        finalresults.append((adv_payoff, error,round(error-adv_payoff,FLAGS.numdecimalplaces), perf, perfmetrics, gen))

    if(searchalg=="GA"):
        alphasaver(AdvInDirC,alphastar[0],gen,gen)
        print('alphastar',alphastar)
        print('bestalpha.fitness.weights[0]',alphastar.fitness.weights[0])
        print('adv_payoff_highest',adv_payoff_highest)
        print('gen',gen)
        print('FLAGS.steplow',FLAGS.steplow)
        print('FLAGS.stephigh',FLAGS.stephigh)
        print('FLAGS.minwidthstartidx',FLAGS.minwidthstartidx)
        print('FLAGS.minwidthenddx',FLAGS.minwidthenddx)
        print('FLAGS.offspringsizefactor',FLAGS.offspringsizefactor)
        print('FLAGS.numgens',FLAGS.numgens)
        print('FLAGS.numalphas',FLAGS.numalphas)
        print('FLAGS.myepsilon',FLAGS.myepsilon)
        print('FLAGS.mylambda',FLAGS.mylambda)
    elif(searchalg=="SA"):
        print('TempMax',TempMax)
        print('TempMin',TempMin)
        print('SampleSize',SampleSize)
        print('ReductionRate',ReductionRate)
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

