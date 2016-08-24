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
    
    
# def generatereports():
#     l = [(1, 0, 1, 0.6547413793103448, {'recall': 0.982535575679172, 'fpr': 0.15983357012380758, 'f1score': 0.6547413793103448, 'precision': 0.49095022624434387, 'tpr': 0.982535575679172}, 0), (0, 0, 1, 0.6576402321083172, {'recall': 0.9836065573770492, 'fpr': 0.2279419784054968, 'f1score': 0.6576402321083172, 'precision': 0.4939467312348668, 'tpr': 0.9836065573770492}, 0), (1.414797432579281, 0.6568338249754179, 0.24203639239613706, 0.6568338249754179, {'recall': 1.0, 'fpr': 0.13718553459119498, 'f1score': 0.6568338249754179, 'precision': 0.4890190336749634, 'tpr': 1.0}, 0), (1.4068787949805888, 0.6489151873767258, 0.24203639239613706, 0.6489151873767258, {'recall': 1.0, 'fpr': 0.1396078431372549, 'f1score': 0.6489151873767258, 'precision': 0.4802919708029197, 'tpr': 1.0}, 1), (1.3459324953773266, 0.6412825651302605, 0.2953500697529339, 0.6412825651302605, {'recall': 1.0, 'fpr': 0.14009000195656426, 'f1score': 0.6412825651302605, 'precision': 0.471976401179941, 'tpr': 1.0}, 2), (1.3085539426665997, 0.6424717999019127, 0.333917857235313, 0.6424717999019127, {'recall': 1.0, 'fpr': 0.1426614481409002, 'f1score': 0.6424717999019127, 'precision': 0.47326589595375723, 'tpr': 1.0}, 3), (1.3141934747726394, 0.6481113320079522, 0.333917857235313, 0.6481113320079522, {'recall': 1.0, 'fpr': 0.1387963144481474, 'f1score': 0.6481113320079522, 'precision': 0.47941176470588237, 'tpr': 1.0}, 4), (1.2617548247472783, 0.6497560975609756, 0.38800127281369745, 0.6497560975609756, {'recall': 1.0, 'fpr': 0.14081192390664837, 'f1score': 0.6497560975609756, 'precision': 0.48121387283236994, 'tpr': 1.0}, 5), (1.2626011368248569, 0.6506024096385542, 0.38800127281369745, 0.6506024096385542, {'recall': 1.0, 'fpr': 0.14221263240486465, 'f1score': 0.6506024096385542, 'precision': 0.48214285714285715, 'tpr': 1.0}, 6), (1.1660467551328677, 0.6547619047619048, 0.48871514962903695, 0.6547619047619048, {'recall': 1.0, 'fpr': 0.13671184443134943, 'f1score': 0.6547619047619048, 'precision': 0.48672566371681414, 'tpr': 1.0}, 7), (1.1576046969380618, 0.6480769230769231, 0.4904722261388612, 0.6480769230769231, {'recall': 1.0, 'fpr': 0.14347314778518228, 'f1score': 0.6480769230769231, 'precision': 0.4793741109530583, 'tpr': 1.0}, 8), (1.2507572896563466, 0.6521739130434783, 0.4014166233871317, 0.6521739130434783, {'recall': 1.0, 'fpr': 0.1381746810598626, 'f1score': 0.6521739130434783, 'precision': 0.4838709677419355, 'tpr': 1.0}, 9), (1.1622014370295546, 0.6526736631684158, 0.4904722261388612, 0.6526736631684158, {'recall': 1.0, 'fpr': 0.1364350215940322, 'f1score': 0.6526736631684158, 'precision': 0.4844213649851632, 'tpr': 1.0}, 10), (1.1487119583746441, 0.6621880998080614, 0.5134761414334172, 0.6621880998080614, {'recall': 1.0, 'fpr': 0.13858267716535433, 'f1score': 0.6621880998080614, 'precision': 0.4949784791965567, 'tpr': 1.0}, 11), (1.1461020238750688, 0.6551226551226551, 0.5090206312475865, 0.6551226551226551, {'recall': 1.0, 'fpr': 0.1408367707719505, 'f1score': 0.6551226551226551, 'precision': 0.4871244635193133, 'tpr': 1.0}, 12), (1.1543651955240672, 0.6633858267716536, 0.5090206312475865, 0.6633858267716536, {'recall': 1.0, 'fpr': 0.1346987002756991, 'f1score': 0.6633858267716536, 'precision': 0.4963181148748159, 'tpr': 1.0}, 13), (1.1455699505708292, 0.6660341555977229, 0.5204642050268937, 0.6660341555977229, {'recall': 1.0, 'fpr': 0.13874655104454078, 'f1score': 0.6660341555977229, 'precision': 0.4992887624466572, 'tpr': 1.0}, 14), (1.1146951781201622, 0.6653771760154739, 0.5506819978953117, 0.6653771760154739, {'recall': 1.0, 'fpr': 0.13635467980295565, 'f1score': 0.6653771760154739, 'precision': 0.4985507246376812, 'tpr': 1.0}, 15), (1.1004470194982732, 0.6534839924670434, 0.5530369729687701, 0.6534839924670434, {'recall': 1.0, 'fpr': 0.14448370632116214, 'f1score': 0.6534839924670434, 'precision': 0.4853146853146853, 'tpr': 1.0}, 16), (1.1034258759528266, 0.6647029945999018, 0.5612771186470751, 0.6647029945999018, {'recall': 1.0, 'fpr': 0.13455476753349094, 'f1score': 0.6647029945999018, 'precision': 0.49779411764705883, 'tpr': 1.0}, 17), (1.0992764121617022, 0.6536585365853659, 0.5543821244236637, 0.6536585365853659, {'recall': 1.0, 'fpr': 0.13940702925584136, 'f1score': 0.6536585365853659, 'precision': 0.4855072463768116, 'tpr': 1.0}, 18), (1.1076420528359932, 0.6552053486150907, 0.5475632957790975, 0.6552053486150907, {'recall': 1.0, 'fpr': 0.14181889609114123, 'f1score': 0.6552053486150907, 'precision': 0.4872159090909091, 'tpr': 1.0}, 19), (0, 0.6552053486150907, 1.6552053486150906, 0.6607258587167855, {'recall': 0.9864537977745524, 'fpr': 0.22557047712632383, 'f1score': 0.6607258587167855, 'precision': 0.4967113276492083, 'tpr': 0.9864537977745524}, 20), (0, 0.6552053486150907, 1.6552053486150906, 0.6608474845262656, {'recall': 0.9876660341555977, 'fpr': 0.23053401769138365, 'f1score': 0.6608474845262656, 'precision': 0.49654185547340807, 'tpr': 0.9876660341555977}, 20)]
#     numgens = 20
#     numalphas = 20
#     myepsilon = 0.0001
#     mylambda = 1
# 
#     l = [(1, 0, 1, 0.6558079725675097, {'recall': 0.9788867562380038, 'fpr': 0.1597440844927389, 'f1score': 0.6558079725675097, 'precision': 0.49307122139864645, 'tpr': 0.9788867562380038}, 0), (0, 0, 1, 0.6569626394953906, {'recall': 0.9849660523763336, 'fpr': 0.22784258148915296, 'f1score': 0.6569626394953906, 'precision': 0.49284154331472946, 'tpr': 0.9849660523763336}, 0), (1.3651282317376676, 0.6470588235294118, 0.28193059179174407, 0.6470588235294118, {'recall': 1.0, 'fpr': 0.14109347442680775, 'f1score': 0.6470588235294118, 'precision': 0.4782608695652174, 'tpr': 1.0}, 0), (1.3770177921928193, 0.6589483839845635, 0.28193059179174407, 0.6589483839845635, {'recall': 1.0, 'fpr': 0.13903638151425762, 'f1score': 0.6589483839845635, 'precision': 0.4913669064748201, 'tpr': 1.0}, 1), (1.3670707027532045, 0.6504312531709792, 0.2833605504177745, 0.6504312531709792, {'recall': 1.0, 'fpr': 0.13517755542475965, 'f1score': 0.6504312531709792, 'precision': 0.48195488721804514, 'tpr': 1.0}, 2), (1.334106515817929, 0.6475531389026199, 0.31344662308469085, 0.6475531389026199, {'recall': 1.0, 'fpr': 0.13974911799294396, 'f1score': 0.6475531389026199, 'precision': 0.4788011695906433, 'tpr': 1.0}, 3), (1.3367129605099304, 0.6523855890944499, 0.3156726285845195, 0.6523855890944499, {'recall': 1.0, 'fpr': 0.14013738959764474, 'f1score': 0.6523855890944499, 'precision': 0.4841040462427746, 'tpr': 1.0}, 4), (1.3295978726282573, 0.6608015640273704, 0.331203691399113, 0.6608015640273704, {'recall': 1.0, 'fpr': 0.136560409287682, 'f1score': 0.6608015640273704, 'precision': 0.49343065693430654, 'tpr': 1.0}, 5), (1.3165120446414962, 0.6477157360406092, 0.331203691399113, 0.6477157360406092, {'recall': 1.0, 'fpr': 0.1360517545579298, 'f1score': 0.6477157360406092, 'precision': 0.47897897897897895, 'tpr': 1.0}, 6), (1.3282022491949466, 0.6594059405940594, 0.331203691399113, 0.6594059405940594, {'recall': 1.0, 'fpr': 0.13532651455546812, 'f1score': 0.6594059405940594, 'precision': 0.4918759231905465, 'tpr': 1.0}, 7), (1.3164966405117453, 0.6477003319108582, 0.331203691399113, 0.6477003319108582, {'recall': 1.0, 'fpr': 0.14560062708210855, 'f1score': 0.6477003319108582, 'precision': 0.4789621318373071, 'tpr': 1.0}, 8), (1.323519509394276, 0.656641604010025, 0.33312209461574915, 0.656641604010025, {'recall': 1.0, 'fpr': 0.13463050314465408, 'f1score': 0.656641604010025, 'precision': 0.48880597014925375, 'tpr': 1.0}, 9), (0, 0.656641604010025, 1.6566416040100251, 0.6645489199491741, {'recall': 0.9840075258701787, 'fpr': 0.22742694538688848, 'f1score': 0.6645489199491741, 'precision': 0.5016786570743406, 'tpr': 0.9840075258701787}, 10), (0, 0.656641604010025, 1.6566416040100251, 0.6633472534532605, {'recall': 0.9833333333333333, 'fpr': 0.22549234135667395, 'f1score': 0.6633472534532605, 'precision': 0.5004847309743092, 'tpr': 0.9833333333333333}, 10)]
#     numgens = 10
#     numalphas = 10
#     myepsilon = 0.0001
#     mylambda = 1

    


if __name__ == '__main__':
    
#     InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/'
#     partitoner(InDir)
#    resizer('/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/101_ObjectCategories_Train/crocodile/')
#    resizer('/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/101_ObjectCategories_Train/crocodile_head/')

#    resizer('/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/101_ObjectCategories_Test/crocodile/')
#    resizer('/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/101_ObjectCategories_Test/crocodile_head/')
        

    InDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/' 
    GameInDir = '/home/aneesh/Documents/AdversarialLearningDatasets/ILSVRC2010/cifar10_data/imagenet2010-batches-bin/'

#     partitoner(InDir)
   
#     resizer(InDir+'TrainSplit/BrownDog/')
#     resizer(InDir+'TrainSplit/BlackDog/')
#   
#     resizer(InDir+'TestSplit/BrownDog/')
#     resizer(InDir+'TestSplit/BlackDog/')

#     resizer(InDir+'TrainSplit/SmallCat/')
#     resizer(InDir+'TestSplit/SmallCat/')

    binarizer(InDir,'TrainSplit/','train.bin')
    copyfile(InDir + 'train.bin', GameInDir + 'train.bin')
     
    binarizer(InDir,'TestSplit/','test.bin')
    copyfile(InDir + 'test.bin', GameInDir + 'test.bin')

