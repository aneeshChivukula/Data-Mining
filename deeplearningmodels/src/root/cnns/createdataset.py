from PIL import Image
import io
import math
import numpy
from os import listdir
import os
from shutil import copyfile
import sys
import pandas as pd
from scipy import stats

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
    
    
def generatereports():
#     rls = [(1, 0, 1, 0.5008010253123999, {'recall': 0.5127952755905512, 'fpr': 0.16724774405250206, 'f1score': 0.5008010253123999, 'precision': 0.48935504070131497, 'tpr': 0.5127952755905512}, 0), (0, 0, 1, 0.6223674655047204, {'recall': 0.8410206084396468, 'fpr': 0.07452678040913335, 'f1score': 0.6223674655047204, 'precision': 0.49394812680115274, 'tpr': 0.8410206084396468}, 0), (1.261150870056856, 0.5171407731582787, 0.25598990310142256, 0.5171407731582787, {'recall': 0.5283159463487332, 'fpr': 0.13661526294978252, 'f1score': 0.5171407731582787, 'precision': 0.5064285714285715, 'tpr': 0.5283159463487332}, 0), (1.2981137407009944, 0.5251798561151079, 0.22706611541411337, 0.5251798561151079, {'recall': 0.5423476968796433, 'fpr': 0.13929560743965175, 'f1score': 0.5251798561151079, 'precision': 0.5090655509065551, 'tpr': 0.5423476968796433}, 1), (1.2981137407009944, 0.5251798561151079, 0.22706611541411337, 0.5251798561151079, {'recall': 0.5423476968796433, 'fpr': 0.13929560743965175, 'f1score': 0.5251798561151079, 'precision': 0.5090655509065551, 'tpr': 0.5423476968796433}, 2), (0, 0.5251798561151079, 1.5251798561151078, 0.6334647079899678, {'recall': 0.8516377649325626, 'fpr': 0.07388199285835742, 'f1score': 0.6334647079899678, 'precision': 0.5042783799201369, 'tpr': 0.8516377649325626}, 3), (0, 0.5251798561151079, 1.5251798561151078, 0.6280575539568345, {'recall': 0.8500486854917235, 'fpr': 0.07474730315127835, 'f1score': 0.6280575539568345, 'precision': 0.4980034227039361, 'tpr': 0.8500486854917235}, 3)]
#     gen = 3
#     numgens = 10
#     numalphas = 20
#     myepsilon = 0.001
#     mylambda = 1
#     final_payoff = 1.2981137407009944

#     rls = [(1, 0, 1, 0.5023211141347848, {'recall': 0.5135842880523731, 'fpr': 0.16654694715238583, 'f1score': 0.5023211141347848, 'precision': 0.49154135338345867, 'tpr': 0.5135842880523731}, 0), (0, 0, 1, 0.6268980477223427, {'recall': 0.8458536585365853, 'fpr': 0.07422505307855626, 'f1score': 0.6268980477223427, 'precision': 0.49798966111430215, 'tpr': 0.8458536585365853}, 0), (1.2612787655263276, 0.5346112886048988, 0.27333252307857125, 0.5346112886048988, {'recall': 0.555719557195572, 'fpr': 0.14053518334985135, 'f1score': 0.5346112886048988, 'precision': 0.5150478796169631, 'tpr': 0.555719557195572}, 0), (1.2548944738611418, 0.5183836912995996, 0.26348921743845777, 0.5183836912995996, {'recall': 0.530156366344006, 'fpr': 0.1368400237294839, 'f1score': 0.5183836912995996, 'precision': 0.5071225071225072, 'tpr': 0.530156366344006}, 1), (1.213784390606654, 0.5161290322580645, 0.3023446416514104, 0.5161290322580645, {'recall': 0.5230312035661219, 'fpr': 0.13415116739216462, 'f1score': 0.5161290322580645, 'precision': 0.5094066570188133, 'tpr': 0.5230312035661219}, 2), (1.1541369357660634, 0.5228758169934641, 0.3687388812274006, 0.5228758169934641, {'recall': 0.5325443786982249, 'fpr': 0.13510301109350237, 'f1score': 0.5228758169934641, 'precision': 0.5135520684736091, 'tpr': 0.5325443786982249}, 3), (1.1541369357660634, 0.5228758169934641, 0.3687388812274006, 0.5228758169934641, {'recall': 0.5325443786982249, 'fpr': 0.13510301109350237, 'f1score': 0.5228758169934641, 'precision': 0.5135520684736091, 'tpr': 0.5325443786982249}, 4), (0, 0.5228758169934641, 1.522875816993464, 0.6276060388209921, {'recall': 0.8533724340175953, 'fpr': 0.07523138320455125, 'f1score': 0.6276060388209921, 'precision': 0.496304718590108, 'tpr': 0.8533724340175953}, 5), (0, 0.5228758169934641, 1.522875816993464, 0.6211135213304411, {'recall': 0.8504950495049505, 'fpr': 0.07608142493638677, 'f1score': 0.6211135213304411, 'precision': 0.4891799544419134, 'tpr': 0.8504950495049505}, 5)]
#     gen = 5
#     numgens = 10
#     numalphas = 20
#     myepsilon = 0.0001
#     mylambda = 1
#     final_payoff = 1.1541369357660634
 
#     rls = [(1, 0, 1, 0.5114878783077167, {'recall': 0.5238558909444986, 'fpr': 0.1662722502315053, 'f1score': 0.5114878783077167, 'precision': 0.4996904024767802, 'tpr': 0.5238558909444986}, 0), (0, 0, 1, 0.6286539155539517, {'recall': 0.8481012658227848, 'fpr': 0.07415272233075682, 'f1score': 0.6286539155539517, 'precision': 0.4994266055045872, 'tpr': 0.8481012658227848}, 0), (0.7453612461976168, 0.05208863058481657, 0.3067273843871997, 0.5208863058481656, {'recall': 0.532293986636971, 'fpr': 0.1363546408074411, 'f1score': 0.5208863058481656, 'precision': 0.5099573257467994, 'tpr': 0.532293986636971}, 0), (0.7379801618702354, 0.05373665480427047, 0.31575649293403496, 0.5373665480427047, {'recall': 0.5527086383601757, 'fpr': 0.13686928883591576, 'f1score': 0.5373665480427047, 'precision': 0.5228531855955678, 'tpr': 0.5527086383601757}, 1), (0.6984292027158623, 0.05024082993701371, 0.35181162722115134, 0.5024082993701371, {'recall': 0.5124716553287982, 'fpr': 0.13748276541264526, 'f1score': 0.5024082993701371, 'precision': 0.49273255813953487, 'tpr': 0.5124716553287982}, 2), (0.6984292027158623, 0.05024082993701371, 0.35181162722115134, 0.5024082993701371, {'recall': 0.5124716553287982, 'fpr': 0.13748276541264526, 'f1score': 0.5024082993701371, 'precision': 0.49273255813953487, 'tpr': 0.5124716553287982}, 3), (0, 0.05024082993701371, 1.0502408299370136, 0.6289803220035778, {'recall': 0.860078277886497, 'fpr': 0.07590422822210902, 'f1score': 0.6289803220035778, 'precision': 0.4957698815566836, 'tpr': 0.860078277886497}, 4), (0, 0.05024082993701371, 1.0502408299370136, 0.6198406951484432, {'recall': 0.8475247524752475, 'fpr': 0.07599660729431722, 'f1score': 0.6198406951484432, 'precision': 0.4885844748858447, 'tpr': 0.8475247524752475}, 4)]
#     gen = 4
#     numgens = 10
#     numalphas = 20
#     myepsilon = 0.0001
#     mylambda = 0.1
#     final_payoff = 0.6984292027158623
 
    rls = [(1, 0, 1, 0.508301404853129, {'recall': 0.5158781594296824, 'fpr': 0.16326950792670372, 'f1score': 0.508301404853129, 'precision': 0.500943989930774, 'tpr': 0.5158781594296824}, 0), (0, 0, 1, 0.6295364714337046, {'recall': 0.8504854368932039, 'fpr': 0.07451146983857264, 'f1score': 0.6295364714337046, 'precision': 0.4997147746719909, 'tpr': 0.8504854368932039}, 0), (1.3159476296722432, 0.5241528478731075, 0.20820521820086424, 0.5241528478731075, {'recall': 0.5401188707280832, 'fpr': 0.13870201820340325, 'f1score': 0.5241528478731075, 'precision': 0.5091036414565826, 'tpr': 0.5401188707280832}, 0), (1.27991153365847, 0.5305826999638075, 0.25067116630533737, 0.5305826999638075, {'recall': 0.535427319211103, 'fpr': 0.1313854104551779, 'f1score': 0.5305826999638075, 'precision': 0.5258249641319943, 'tpr': 0.535427319211103}, 1), (1.27991153365847, 0.5305826999638075, 0.25067116630533737, 0.5305826999638075, {'recall': 0.535427319211103, 'fpr': 0.1313854104551779, 'f1score': 0.5305826999638075, 'precision': 0.5258249641319943, 'tpr': 0.535427319211103}, 2), (0, 0.5305826999638075, 1.5305826999638075, 0.6349319971367215, {'recall': 0.8553519768563163, 'fpr': 0.07396072430502423, 'f1score': 0.6349319971367215, 'precision': 0.5048377916903813, 'tpr': 0.8553519768563163}, 3), (0, 0.5305826999638075, 1.5305826999638075, 0.6240928882438317, {'recall': 0.8414872798434442, 'fpr': 0.07420614705382918, 'f1score': 0.6240928882438317, 'precision': 0.49596309111880044, 'tpr': 0.8414872798434442}, 3)]
    gen = 3
    numgens = 20
    numalphas = 100
    myepsilon = 0.0001
    mylambda = 1
    final_payoff = 1.27991153365847
    
    results = {}

    rls[0][4]['payoff'] = rls[0][0]
    rls[0][4]['error'] = rls[0][1]
    rls[0][4]['norm'] = rls[0][2]
    results['training error on original training data and original testing data'] = rls[0][4]

    rls[1][4]['payoff'] = rls[1][0]
    rls[1][4]['error'] = rls[1][1]
    rls[1][4]['norm'] = rls[1][2]    
    results['testing error on original training data and original testing data'] = rls[1][4]

    rls[-2][4]['payoff'] = rls[-2][0]
    rls[-2][4]['error'] = rls[-2][1]
    rls[-2][4]['norm'] = rls[-2][2]    
    results['testing error on manipulated training data and manipulated testing data'] = rls[-2][4]

    rls[-1][4]['payoff'] = rls[-1][0]
    rls[-1][4]['error'] = rls[-1][1]
    rls[-1][4]['norm'] = rls[-1][2]    
    results['testing error on original training data and manipulated testing data'] = rls[-1][4]

    print(results)
    print('numgens',numgens)
    print('numalphas',numalphas)
    print('myepsilon',myepsilon)
    print('mylambda',mylambda)
    print('final_payoff',final_payoff)


 
#     l = [(1, 0, 1, 0.6558079725675097, {'recall': 0.9788867562380038, 'fpr': 0.1597440844927389, 'f1score': 0.6558079725675097, 'precision': 0.49307122139864645, 'tpr': 0.9788867562380038}, 0), (0, 0, 1, 0.6569626394953906, {'recall': 0.9849660523763336, 'fpr': 0.22784258148915296, 'f1score': 0.6569626394953906, 'precision': 0.49284154331472946, 'tpr': 0.9849660523763336}, 0), (1.3651282317376676, 0.6470588235294118, 0.28193059179174407, 0.6470588235294118, {'recall': 1.0, 'fpr': 0.14109347442680775, 'f1score': 0.6470588235294118, 'precision': 0.4782608695652174, 'tpr': 1.0}, 0), (1.3770177921928193, 0.6589483839845635, 0.28193059179174407, 0.6589483839845635, {'recall': 1.0, 'fpr': 0.13903638151425762, 'f1score': 0.6589483839845635, 'precision': 0.4913669064748201, 'tpr': 1.0}, 1), (1.3670707027532045, 0.6504312531709792, 0.2833605504177745, 0.6504312531709792, {'recall': 1.0, 'fpr': 0.13517755542475965, 'f1score': 0.6504312531709792, 'precision': 0.48195488721804514, 'tpr': 1.0}, 2), (1.334106515817929, 0.6475531389026199, 0.31344662308469085, 0.6475531389026199, {'recall': 1.0, 'fpr': 0.13974911799294396, 'f1score': 0.6475531389026199, 'precision': 0.4788011695906433, 'tpr': 1.0}, 3), (1.3367129605099304, 0.6523855890944499, 0.3156726285845195, 0.6523855890944499, {'recall': 1.0, 'fpr': 0.14013738959764474, 'f1score': 0.6523855890944499, 'precision': 0.4841040462427746, 'tpr': 1.0}, 4), (1.3295978726282573, 0.6608015640273704, 0.331203691399113, 0.6608015640273704, {'recall': 1.0, 'fpr': 0.136560409287682, 'f1score': 0.6608015640273704, 'precision': 0.49343065693430654, 'tpr': 1.0}, 5), (1.3165120446414962, 0.6477157360406092, 0.331203691399113, 0.6477157360406092, {'recall': 1.0, 'fpr': 0.1360517545579298, 'f1score': 0.6477157360406092, 'precision': 0.47897897897897895, 'tpr': 1.0}, 6), (1.3282022491949466, 0.6594059405940594, 0.331203691399113, 0.6594059405940594, {'recall': 1.0, 'fpr': 0.13532651455546812, 'f1score': 0.6594059405940594, 'precision': 0.4918759231905465, 'tpr': 1.0}, 7), (1.3164966405117453, 0.6477003319108582, 0.331203691399113, 0.6477003319108582, {'recall': 1.0, 'fpr': 0.14560062708210855, 'f1score': 0.6477003319108582, 'precision': 0.4789621318373071, 'tpr': 1.0}, 8), (1.323519509394276, 0.656641604010025, 0.33312209461574915, 0.656641604010025, {'recall': 1.0, 'fpr': 0.13463050314465408, 'f1score': 0.656641604010025, 'precision': 0.48880597014925375, 'tpr': 1.0}, 9), (0, 0.656641604010025, 1.6566416040100251, 0.6645489199491741, {'recall': 0.9840075258701787, 'fpr': 0.22742694538688848, 'f1score': 0.6645489199491741, 'precision': 0.5016786570743406, 'tpr': 0.9840075258701787}, 10), (0, 0.656641604010025, 1.6566416040100251, 0.6633472534532605, {'recall': 0.9833333333333333, 'fpr': 0.22549234135667395, 'f1score': 0.6633472534532605, 'precision': 0.5004847309743092, 'tpr': 0.9833333333333333}, 10)]
#     numgens = 10
#     numalphas = 10
#     myepsilon = 0.0001
#     mylambda = 1

    
def ttest():
    records = []
    record = {'testing error on original training data and original testing data': {'recall': 0.8410206084396468, 'error': 0, 'precision': 0.49394812680115274, 'tpr': 0.8410206084396468, 'fpr': 0.07452678040913335, 'f1score': 0.6223674655047204, 'norm': 1, 'payoff': 0}, 'testing error on original training data and manipulated testing data': {'recall': 0.8500486854917235, 'error': 0.5251798561151079, 'precision': 0.4980034227039361, 'tpr': 0.8500486854917235, 'fpr': 0.07474730315127835, 'f1score': 0.6280575539568345, 'norm': 1.5251798561151078, 'payoff': 0}, 'training error on original training data and original testing data': {'recall': 0.5127952755905512, 'error': 0, 'precision': 0.48935504070131497, 'tpr': 0.5127952755905512, 'fpr': 0.16724774405250206, 'f1score': 0.5008010253123999, 'norm': 1, 'payoff': 1}, 'testing error on manipulated training data and manipulated testing data': {'recall': 0.8516377649325626, 'error': 0.5251798561151079, 'precision': 0.5042783799201369, 'tpr': 0.8516377649325626, 'fpr': 0.07388199285835742, 'f1score': 0.6334647079899678, 'norm': 1.5251798561151078, 'payoff': 0}}
    records.append(record)
    record = {'testing error on original training data and original testing data': {'recall': 0.8458536585365853, 'error': 0, 'precision': 0.49798966111430215, 'tpr': 0.8458536585365853, 'fpr': 0.07422505307855626, 'f1score': 0.6268980477223427, 'norm': 1, 'payoff': 0}, 'testing error on original training data and manipulated testing data': {'recall': 0.8504950495049505, 'error': 0.5228758169934641, 'precision': 0.4891799544419134, 'tpr': 0.8504950495049505, 'fpr': 0.07608142493638677, 'f1score': 0.6211135213304411, 'norm': 1.522875816993464, 'payoff': 0}, 'training error on original training data and original testing data': {'recall': 0.5135842880523731, 'error': 0, 'precision': 0.49154135338345867, 'tpr': 0.5135842880523731, 'fpr': 0.16654694715238583, 'f1score': 0.5023211141347848, 'norm': 1, 'payoff': 1}, 'testing error on manipulated training data and manipulated testing data': {'recall': 0.8533724340175953, 'error': 0.5228758169934641, 'precision': 0.496304718590108, 'tpr': 0.8533724340175953, 'fpr': 0.07523138320455125, 'f1score': 0.6276060388209921, 'norm': 1.522875816993464, 'payoff': 0}}
    records.append(record)
    record = {'testing error on original training data and original testing data': {'recall': 0.8481012658227848, 'error': 0, 'precision': 0.4994266055045872, 'tpr': 0.8481012658227848, 'fpr': 0.07415272233075682, 'f1score': 0.6286539155539517, 'norm': 1, 'payoff': 0}, 'testing error on original training data and manipulated testing data': {'recall': 0.8475247524752475, 'error': 0.05024082993701371, 'precision': 0.4885844748858447, 'tpr': 0.8475247524752475, 'fpr': 0.07599660729431722, 'f1score': 0.6198406951484432, 'norm': 1.0502408299370136, 'payoff': 0}, 'training error on original training data and original testing data': {'recall': 0.5238558909444986, 'error': 0, 'precision': 0.4996904024767802, 'tpr': 0.5238558909444986, 'fpr': 0.1662722502315053, 'f1score': 0.5114878783077167, 'norm': 1, 'payoff': 1}, 'testing error on manipulated training data and manipulated testing data': {'recall': 0.860078277886497, 'error': 0.05024082993701371, 'precision': 0.4957698815566836, 'tpr': 0.860078277886497, 'fpr': 0.07590422822210902, 'f1score': 0.6289803220035778, 'norm': 1.0502408299370136, 'payoff': 0}}
    records.append(record)
    record = {'testing error on original training data and original testing data': {'recall': 0.8504854368932039, 'error': 0, 'precision': 0.4997147746719909, 'tpr': 0.8504854368932039, 'fpr': 0.07451146983857264, 'f1score': 0.6295364714337046, 'norm': 1, 'payoff': 0}, 'testing error on original training data and manipulated testing data': {'recall': 0.8414872798434442, 'error': 0.5305826999638075, 'precision': 0.49596309111880044, 'tpr': 0.8414872798434442, 'fpr': 0.07420614705382918, 'f1score': 0.6240928882438317, 'norm': 1.5305826999638075, 'payoff': 0}, 'training error on original training data and original testing data': {'recall': 0.5158781594296824, 'error': 0, 'precision': 0.500943989930774, 'tpr': 0.5158781594296824, 'fpr': 0.16326950792670372, 'f1score': 0.508301404853129, 'norm': 1, 'payoff': 1}, 'testing error on manipulated training data and manipulated testing data': {'recall': 0.8553519768563163, 'error': 0.5305826999638075, 'precision': 0.5048377916903813, 'tpr': 0.8553519768563163, 'fpr': 0.07396072430502423, 'f1score': 0.6349319971367215, 'norm': 1.5305826999638075, 'payoff': 0}}
    records.append(record)
    
    l1 = []
    l2 = []
    l3 = []
    
    for record in records:
        l1.append(record['testing error on original training data and original testing data']['precision'])
        l2.append(record['testing error on original training data and manipulated testing data']['precision'])
        l3.append(record['testing error on manipulated training data and manipulated testing data']['precision'])
#         l.append(record['training error on original training data and original testing data']['precision'])
        

    ttest=stats.ttest_ind(l1,l2,equal_var=True)
    print 't-statistic independent = %6.3f pvalue = %6.4f' % ttest
#     ttest=stats.ttest_rel(l1,l2)
#     print 't-statistic dependent = %6.3f pvalue = %6.4f' % ttest
    print('l1',l1)
    print('l2',l2)
    print('l3',l3)
    print 'Need to have length of l to be 20'
    

    
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

#     binarizer(InDir,'TrainSplit/','train.bin')
#     copyfile(InDir + 'train.bin', GameInDir + 'train.bin')
#      
#     binarizer(InDir,'TestSplit/','test.bin')
#     copyfile(InDir + 'test.bin', GameInDir + 'test.bin')
#     generatereports()
    
    ttest()
