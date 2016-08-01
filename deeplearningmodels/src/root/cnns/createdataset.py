from PIL import Image
import io
import math
import numpy
from os import listdir
import os
from shutil import copyfile
import sys


width = 32
height = 32
length = 3073

# width = 224
# height = 224

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
        
        
def colourtograyscale(InDir,OutDir):
    os.chdir(InDir)
    for f in listdir(InDir):
        img = Image.open(f)
        # http://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
        # http://stackoverflow.com/questions/9506841/using-python-pil-to-turn-a-rgb-image-into-a-pure-black-and-white-image
        


def binarizer(CurrDir,ValDir,OutFile): 
    # resizer and binarizer have same directory
    # Pixel value is in the form (R,G,B) where R,G,B belongs in the range 0,255
    ind = 0
    L = []
    os.chdir(CurrDir)
    binfile = open(OutFile, 'wb',)
    os.chdir(CurrDir + ValDir)
    ls = listdir(CurrDir + ValDir)
    ls.sort()
    for d in ls:
        os.chdir(CurrDir + ValDir + d)
        for f in listdir(CurrDir + ValDir + d):
            img = Image.open(f)
            l = numpy.insert(numpy.array(img.getdata()).flatten(order='F'),0, ind)
            if(len(l) == length):
                L.append(l)
        ind = ind + 1
    
    print(len(L)*length)
    print(len(numpy.concatenate(L)))
    
    numpy.concatenate(L).astype('int16').tofile(binfile)
#     file.write(numpy.concatenate(L).astype('int16'))
    binfile.close()

    

if __name__ == '__main__':
    
#     InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/'
#     partitoner(InDir)
#    resizer('/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/101_ObjectCategories_Train/crocodile/')
#    resizer('/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/101_ObjectCategories_Train/crocodile_head/')

#    resizer('/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/101_ObjectCategories_Test/crocodile/')
#    resizer('/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/101_ObjectCategories_Test/crocodile_head/')
        

    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/' 
#     partitoner(InDir)
   
#     resizer(InDir+'TrainSplit/BrownDog/')
#     resizer(InDir+'TrainSplit/BlackDog/')
#   
#     resizer(InDir+'TestSplit/BrownDog/')
#     resizer(InDir+'TestSplit/BlackDog/')

    binarizer(InDir,'TrainSplit/','train.bin')
    binarizer(InDir,'TestSplit/','test.bin')


