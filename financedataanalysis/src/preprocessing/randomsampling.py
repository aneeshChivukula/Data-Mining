from datetime import datetime, date, time
import sys
import random
import linecache
import cPickle
from statsmodels.iolib.table import Row
import numpy as np
import math
from os import listdir


def createpositivenegativepartitions(InFile,PositiveClassFile,NegativeClassFile,Sampletimestampsfile,Samplepricesfile):

    startdate = date(2012, 9, 03)
    starttime = time(9, 25, 0, 1)
    startdatetime = datetime.combine(startdate, starttime)    
    
    fp = open(PositiveClassFile, 'w')
    fn = open(NegativeClassFile, 'w')
    numpositivelines = 0
    numnegativelines = 0
    
    sampletimestamps = list()
    sampleprices = list()
    
    with open(InFile) as fileobject:
        for line in fileobject:
            row = line.rstrip('\n').split(',')
            d = row[0].split('/')
            t = row[1].split(':')
            ts = t[-1].split('.')

            currdate = date(int(d[2]), int(d[1]), int(d[0]))
            currtime = time(int(t[0]), int(t[1]), int(ts[0]), int(ts[1]))

            currdatetime = datetime.combine(currdate, currtime)
            timesince = currdatetime - startdatetime
            minutessince = int(timesince.total_seconds() / 60)  
            secondssince = int(timesince.total_seconds())

            if(row[5] == ''):
                fn.write(str(secondssince)+','+row[2]+','+row[3]+','+row[4]+','+row[5]+'\n')
                numnegativelines = numnegativelines + 1
            else:            
                fp.write(str(secondssince)+','+row[2]+','+row[3]+','+row[4]+','+row[5]+'\n')
                numpositivelines = numpositivelines + 1
                
            sampletimestamps.append(secondssince)
            sampleprices.append(row[4])
    
    cPickle.dump(sampletimestamps, open(Sampletimestampsfile, 'wb')) 
    cPickle.dump(sampleprices, open(Samplepricesfile, 'wb')) 
    print 'numpositivelines',numpositivelines
    print 'numnegativelines',numnegativelines
    print 'done'


def createpositivesample(PositiveClassFile,PositiveSampleFile,Positivetimestampsfile,numpositivelines,positivesamplesize,LabelledSampleFile,InFile,Sampletimestampsfile,Samplepricesfile,windowsize):
    sampletimestamps = cPickle.load(open(Sampletimestampsfile, 'rb'))
    sampleprices = cPickle.load(open(Samplepricesfile, 'rb'))
    
    fs = open(LabelledSampleFile, 'a')
    fp = open(PositiveSampleFile, 'w')
    positivesampletimestamps = list()

    selectedindices = []
    numsamples = 0
    while(numsamples <= positivesamplesize):
        curridx = random.randint(0,numpositivelines)
        if(curridx not in selectedindices):
            selectedindices.append(curridx)
            
            line = linecache.getline(PositiveClassFile, curridx)
            
            splitline = line.split(',')
            currtimestamp = int(splitline[0])
            currprice = splitline[3]
            positivesampletimestamps.append(currtimestamp)

            processedtimestamps = []
            processedtimestamps.append(currtimestamp)
            sampleline = ""
            
            for xidx,x in enumerate(sampletimestamps): 
            # Reduce number of comparisons by indexing presorted timestampped database, binary search partitioning and multiprocessing loops over cpus. 
                # Convert forloop into while loop and use multiprocessing to speedup this loop. Then use same loop to create negative samples. 
                # Execute using server memory.
                if(x not in processedtimestamps and (currtimestamp - windowsize < x < currtimestamp) and len(processedtimestamps) != 59):
                    processedtimestamps.append(x)
                    inprice = sampleprices[xidx]
                    sampleline += inprice + ","
    
            if(len(processedtimestamps) == 59):
                fp.write(line)
                sampleline += currprice + ","
                fs.write(sampleline + "P" + '\n')
                numsamples = numsamples + 1
                print('numsamples',numsamples)
                
    
    fp.close()
    fs.close()
    cPickle.dump(positivesampletimestamps, open(Positivetimestampsfile, 'wb')) 
    print 'done'
    
    
    
def createnegativesample(PositiveClassFile,NegativeClassFile,NegativeSampleFile,Positivetimestampsfile,numnegativelines,negativesamplesize,windowsize,LabelledSampleFile,InFile,Sampletimestampsfile):
    
    fs = open(LabelledSampleFile, 'a')
    fn = open(NegativeSampleFile, 'w')

    positivesampletimestamps = cPickle.load(open(Positivetimestampsfile, 'rb'))
    positivesampletimestampsset = sorted(set(positivesampletimestamps))
    discardedindices = []
    selectedindices = []
    
    while len(selectedindices) <= negativesamplesize:
        curridx = random.randint(0,numnegativelines)
        if(curridx not in discardedindices and curridx not in selectedindices):
            line = linecache.getline(NegativeClassFile, curridx)
            currtimestamp = int(line.rstrip().split(',')[0])


            xidx = 0
            looping = True
            maxxidx = len(positivesampletimestampsset)
#             matchingpositives = []
            
            while looping and xidx < maxxidx:
                x = positivesampletimestampsset[xidx]
#             for x in positivesampletimestampsset:
                if(currtimestamp - windowsize < x < currtimestamp + windowsize): 
                    # Check searching algorithms if loop is too slow. Also need only one matchingpositives to exit loop.
#                     matchingpositives.append(x)
                    looping = False
                else:
                    xidx += 1
            
#             if(len(matchingpositives) == 0):
            if(looping):
                fn.write(line)
#                 print("Found: ",curridx)
                selectedindices.append(curridx)
            else:
                discardedindices.append(curridx)
        
    fn.close()
    fs.close()
    print("Done")

def createlabelleddatasample(PositiveSampleFile,NegativeSampleFile,InFile,LabelledSampleFile,windowsize):
    with open(PositiveSampleFile) as fileobject:
           for line in fileobject:    
               currtimestamp = line.rstrip('\n').split(',')[0]
               inline = linecache.getline(InFile, currtimestamp)
               
               
def createfilteredpositivenegativepartitions(InFile,PositiveClassFile,NegativeClassFile, Sampletimestampsfile,Positivetimestampsfile):

    fp = open(PositiveClassFile, 'w')
    fn = open(NegativeClassFile, 'w')
    
    positivesize = 0
    negativesize = 0
    
    sampletimestamps = []
    positivesampletimestamps = []
    
    with open(InFile) as fileobject:
        for line in fileobject:
            row = line.rstrip('\n').split(',')
            sampletimestamps.append(row[0])
            
            if(row[2] == 'P'):
                fp.write(line)
                positivesampletimestamps.append(row[0])
                positivesize = positivesize + 1
            else:
                fn.write(line)
                negativesize = negativesize + 1

    cPickle.dump(sampletimestamps, open(Sampletimestampsfile, 'wb')) 
    cPickle.dump(positivesampletimestamps, open(Positivetimestampsfile, 'wb'))
    print 'done'
    return (positivesize,negativesize)
    
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
#     return array[idx],idx
    return idx
    
def createfilteredsample(InFile,PositiveClassFile,PositiveSampleFile,LabelledSampleFile,positivesamplesize,negativesamplesize,numpositivelines,numnegativelines,windowsize,Sampletimestampsfile,Positivetimestampsfile,NegativeClassFile,NegativeSampleFile):
    inlines = open(InFile, 'r').readlines()
    
    fs = open(LabelledSampleFile, 'a')

    sampletimestamps = np.array(cPickle.load(open(Sampletimestampsfile, 'rb')),dtype=np.float32)
    
    selectedindices = []
     
    numsamples = 0
     
    while(numsamples <= positivesamplesize):
        curridx = random.randint(0,numpositivelines)
        if(curridx not in selectedindices):
            selectedindices.append(curridx)
             
            line = linecache.getline(PositiveClassFile, curridx)
             
            if(line):
                splitline = line.split(',')
                searchtimestamp = float(splitline[0])
                 
                sampleline = ""
                 
                for d in xrange(1,windowsize+1):
                    curridx = find_nearest(sampletimestamps, searchtimestamp-(windowsize/d))
                    splitline = inlines[curridx].rstrip('\n').split(',')
                     
                    sampleline += splitline[1] + ","
                 
                fs.write(sampleline + "P" + '\n')
                 
                numsamples = numsamples + 1
            


            
    numsamples = 0
    selectedindices = []
    discardedindices = []
    positivesampletimestamps = np.array(cPickle.load(open(Positivetimestampsfile, 'rb')),dtype=np.float32)
    
    
    while(numsamples <= negativesamplesize):
        curridx = random.randint(0,numnegativelines)
        if(curridx not in discardedindices and curridx not in selectedindices):
            line = linecache.getline(NegativeClassFile, curridx)

            if(line):
                splitline = line.split(',')
                searchtimestamp = float(splitline[0])

                sampleline = ""


                if(sum((positivesampletimestamps > searchtimestamp-windowsize) & (positivesampletimestamps < searchtimestamp)) != 0):
                    discardedindices.append(curridx)
                else:
                    selectedindices.append(curridx)

                    sampleline = ""
                    for d in xrange(1,windowsize+1):
                        curridx = find_nearest(sampletimestamps, searchtimestamp-(windowsize/d))
                        splitline = inlines[curridx].rstrip('\n').split(',')
                        sampleline += splitline[1] + ","
                    
                    fs.write(sampleline + "N" + '\n')

                    numsamples = numsamples + 1

    
    print("check negatives")
    sys.exit()














if __name__ == '__main__':
    LabelledSampleFile = '/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/sample.csv'
    fs = open(LabelledSampleFile, 'w')
    fs.close()

    InputsDir = "/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/inputs/"
    ClassesDir = "/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/labelledinputs/"
    inputs = listdir(InputsDir)
    inputs.sort()
    
    windowsize = 10*60
    
    for id in xrange(0,len(inputs)):
        
        InFile = InputsDir + inputs[id]
        
        inputfilename = inputs[id].rstrip('.csv')
        PositiveClassFile = ClassesDir + inputfilename + '_positiveclass.csv'
        NegativeClassFile = ClassesDir + inputfilename + '_negativeclass.csv'
        Sampletimestampsfile = ClassesDir + 'pickledsampletimestamps.csv'
        Positivetimestampsfile = ClassesDir + 'pickledpositivetimestamps.csv'
        PositiveSampleFile = ClassesDir + inputfilename + '_positivesample.csv'
        NegativeSampleFile = ClassesDir + inputfilename + '_negativesample.csv'
        
        (numpositivelines,numnegativelines) = createfilteredpositivenegativepartitions(InFile,PositiveClassFile,NegativeClassFile, Sampletimestampsfile,Positivetimestampsfile)
        positivesamplesize = numpositivelines
        negativesamplesize = numpositivelines
        
        
        print(InFile,PositiveClassFile,PositiveSampleFile,LabelledSampleFile,positivesamplesize,negativesamplesize,numpositivelines,numnegativelines,windowsize,Sampletimestampsfile,Positivetimestampsfile,NegativeClassFile,NegativeSampleFile)
        
        createfilteredsample(InFile,PositiveClassFile,PositiveSampleFile,LabelledSampleFile,positivesamplesize,negativesamplesize,numpositivelines,numnegativelines,windowsize,Sampletimestampsfile,Positivetimestampsfile,NegativeClassFile,NegativeSampleFile)

    print 'Completed sampling across all input files'
    sys.exit()
        

#     InFile = '/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/threemonth_sample_demomarket_1000seriesalerts.csv' 
    InFile = '/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/input.csv' 
    
    LabelledSampleFile = '/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/sample.csv' 
    PositiveClassFile = '/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/threemonth_sample_demomarket_1000seriesalerts_positiveclass.csv' 
    NegativeClassFile = '/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/threemonth_sample_demomarket_1000seriesalerts_negativeclass.csv' 
    Positivetimestampsfile = '/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/pickledpositivetimestamps.csv'
    Sampletimestampsfile = '/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/pickledsampletimestamps.csv'
    Samplepricesfile = '/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/pickledsampleprices.csv'
    
    
    windowsize = 60*60
    numpositivelines = 67
    positivesamplesize = 50
    PositiveSampleFile = '/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/positivesample.csv' 
    numnegativelines = 30109
    negativesamplesize = 50
    NegativeSampleFile = '/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/negativesample.csv' 

    fs = open(LabelledSampleFile, 'w')
    fs.close()
    
#     createfilteredpositivenegativepartitions(InFile,LabelledSampleFile,PositiveClassFile,NegativeClassFile, Sampletimestampsfile,Positivetimestampsfile)
    
    
    createfilteredsample(InFile,PositiveClassFile,PositiveSampleFile,LabelledSampleFile,positivesamplesize,negativesamplesize,numpositivelines,numnegativelines,windowsize,Sampletimestampsfile,Positivetimestampsfile,NegativeClassFile,NegativeSampleFile)
    
    
    
    
    
#     createpositivenegativepartitions(InFile,PositiveClassFile,NegativeClassFile,Sampletimestampsfile,Samplepricesfile)
#     createpositivesample(PositiveClassFile,PositiveSampleFile,Positivetimestampsfile,numpositivelines,positivesamplesize,LabelledSampleFile,InFile,Sampletimestampsfile,Samplepricesfile,windowsize)
#     createnegativesample(PositiveClassFile,NegativeClassFile,NegativeSampleFile,Positivetimestampsfile,numnegativelines,negativesamplesize,windowsize,LabelledSampleFile,InFile,Sampletimestampsfile)

# InFile metadata :
# "timestampdiff","price","binarylabels"
# date,time,instrument,transactionid,price,label
