from datetime import datetime, date, time
import sys
import random
import linecache
import cPickle

def createpositivenegativepartitions(InFile,PositiveClassFile,NegativeClassFile,Sampletimestampsfile):

    startdate = date(2012, 9, 03)
    starttime = time(9, 25, 0, 1)
    startdatetime = datetime.combine(startdate, starttime)    
    
    fp = open(PositiveClassFile, 'w')
    fn = open(NegativeClassFile, 'w')
    numpositivelines = 0
    numnegativelines = 0
    
    sampletimestamps = list()
    
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
    
    cPickle.dump(sampletimestamps, open(Sampletimestampsfile, 'wb')) 
    print 'numpositivelines',numpositivelines
    print 'numnegativelines',numnegativelines
    print 'done'


def createpositivesample(PositiveClassFile,PositiveSampleFile,Positivetimestampsfile,numpositivelines,positivesamplesize,LabelledSampleFile,InFile,Sampletimestampsfile,windowsize):
    sampletimestamps = cPickle.load(open(Sampletimestampsfile, 'rb'))
    
#     print(sampletimestamps)
    
    fs = open(LabelledSampleFile, 'a')

    fp = open(PositiveSampleFile, 'w')
    positivesampletimestamps = list()

    selectedindices = []

    while(len(selectedindices) <= positivesamplesize):
        curridx = random.randint(0,numpositivelines)
        if(curridx not in selectedindices):
            selectedindices.append(curridx)
            
            line = linecache.getline(PositiveClassFile, curridx)
            fp.write(line)
            
            splitline = line.split(',')
            currtimestamp = int(splitline[0])
            currprice = splitline[3]
            positivesampletimestamps.append(currtimestamp)

            processedtimestamps = []
            processedtimestamps.append(currtimestamp)
            sampleline = ""
    
            for x in sampletimestamps:
                if(x not in processedtimestamps and (currtimestamp - windowsize < x < currtimestamp) and len(processedtimestamps) <= 60):
                    inline = linecache.getline(InFile, x) # Taking the first available price value for x
                    inprice = inline.split(',')[4] 
                    processedtimestamps.append(x)
                    sampleline += inprice + ","
    
            if(len(processedtimestamps) == 60):
                sampleline += currprice + ","
                fs.write(sampleline + "P" + '\n')
    
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

            matchingpositives = []
            for x in positivesampletimestampsset:
                if(currtimestamp - windowsize < x < currtimestamp + windowsize): # Check searching algorithms if loop is too slow. Also need only one matchingpositives to exit loop.
                    matchingpositives.append(x)
            print(curridx,matchingpositives)
            
            if(len(matchingpositives) == 0):
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
               
               
               

if __name__ == '__main__':
    InFile = '/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/threemonth_sample_demomarket_1000seriesalerts.csv' 
    LabelledSampleFile = '/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/sample.csv' 
    PositiveClassFile = '/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/threemonth_sample_demomarket_1000seriesalerts_positiveclass.csv' 
    NegativeClassFile = '/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/threemonth_sample_demomarket_1000seriesalerts_negativeclass.csv' 
    Positivetimestampsfile = '/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/pickledpositivetimestamps.csv'
    Sampletimestampsfile = '/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/pickledsampletimestamps.csv'
    windowsize = 60
#     createpositivenegativepartitions(InFile,PositiveClassFile,NegativeClassFile,Sampletimestampsfile)


    fs = open(LabelledSampleFile, 'w')
    fs.close()


    numpositivelines = 48596
    positivesamplesize = 10000
    PositiveSampleFile = '/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/positivesample.csv' 
#     createpositivesample(PositiveClassFile,PositiveSampleFile,Positivetimestampsfile,numpositivelines,positivesamplesize,LabelledSampleFile,InFile,Sampletimestampsfile,windowsize)
    
    numnegativelines = 10745743
    negativesamplesize = 10000
    NegativeSampleFile = '/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/negativesample.csv' 
    createnegativesample(PositiveClassFile,NegativeClassFile,NegativeSampleFile,Positivetimestampsfile,numnegativelines,negativesamplesize,windowsize,LabelledSampleFile,InFile,Sampletimestampsfile)

