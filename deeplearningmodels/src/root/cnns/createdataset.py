import os
import math
import sys
from os import listdir
from PIL import Image
from shutil import copyfile

width = 224
height = 224

#width = 300
#height = 100

def partitoner(Dir):
    os.chdir(Dir)
    
    InDir = Dir + 'LabelledData/'
    classdirs = ['BrownDog/','BlackDog/']

    TrainOutDir = Dir + 'TrainSplit/'
    TestOutDir = Dir + 'TestSplit/'

    for dir in classdirs:
        files = listdir(InDir + dir)
        
        splitind = int(math.floor(0.8*len(files)))
        trainfiles = files[0:splitind]
        testfiles = files[splitind:]
    
        for file in trainfiles:
            copyfile(InDir+dir+file, TrainOutDir+dir+file)

        for file in testfiles:
            copyfile(InDir+dir+file, TestOutDir+dir+file)


def resizer(CurrDir):
    os.chdir(CurrDir)
    for f in listdir(CurrDir):
        
        img = Image.open(f)
        print(img.size)
        image = img.resize((width,height), Image.ANTIALIAS)
        image.save(CurrDir + f)
        print(img.size)



if __name__ == '__main__':
    
#     InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/'
#     partitoner(InDir)
#    resizer('/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/101_ObjectCategories_Train/crocodile/')
#    resizer('/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/101_ObjectCategories_Train/crocodile_head/')

#    resizer('/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/101_ObjectCategories_Test/crocodile/')
#    resizer('/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/101_ObjectCategories_Test/crocodile_head/')
        

#     InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/' 
#     partitoner(InDir)
 
    resizer('/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TrainSplit/BrownDog/')
    resizer('/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TrainSplit/BlackDog/')

    resizer('/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TestSplit/BrownDog/')
    resizer('/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/TestSplit/BlackDog/')
