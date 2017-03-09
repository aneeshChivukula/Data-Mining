
setwd("/home/achivuku/Documents/BigLearningDatasets")
mydata = read.csv("threemonth_sample_demomarket_1000seriesalerts.csv")
head(mydata)
myvar = mydata$instrument
sort(table(myvar),decreasing=TRUE)[1:3]
sort(table(myvar),decreasing=FALSE)[1:1000]
SEC0000431, SEC0000322, SEC0000271

myvar[myvar=='SEC0000271']
mydata[myvar=='SEC0000271',]
head(mydata[myvar=='SEC0000271',])
tail(mydata[myvar=='SEC0000271',])

mydata2 = mydata[myvar=='SEC0000271',]
myvar2 = mydata2$label

nrow(mydata2)
mydata2test = mydata2[1:100,]

timestamp = as.POSIXct(paste(mydata2test$date, mydata2test$time), format="%d/%m/%Y %H:%M:%OS")
timestamp = as.POSIXct(paste(mydata2$date, mydata2$time), format="%d/%m/%Y %H:%M:%OS")
mydata2$timestamp = timestamp

discard milliseconds

difftime("2012-09-03 09:30:06", "2012-09-03 09:30:01 AEST", unit="weeks")
difftime("2012-09-03 09:30:06", "2012-09-03 09:30:01 AEST", unit="secs")

difftime(mydata2[2,]$timestamp,mydata2[1,]$timestamp)
difftime(mydata2[2,]$timestamp,"2012-09-03 09:30:01 AEST")
mydata2[1,]$timestamp : "2012-09-03 09:30:01 AEST"

timestampdiff <- 0
for(i in 1:nrow(mydata2)) {
timestampdiff[i] <- difftime(mydata2[i,]$timestamp,"2012-09-03 09:30:01 AEST",unit="secs")
}

mydata2$timestampdiff = timestampdiff
head(mydata2)
require(ggplot2)
df = mydata2[,c("timestampdiff","price")]
ggplot( data = df, aes( timestampdiff, price )) + geom_line() 

Must use java script to plot data labels and data features in time series. Otherwise use separate standard plots.
http://www.highcharts.com/demo/line-labels

> nrow(mydata2)
[1] 30176

emptylabels = mydata2$label==''

> length(emptylabels[emptylabels==TRUE]) 
[1] 30109
> 
> length(emptylabels[emptylabels==FALSE]) 
[1] 67
> 

mydata2 = mydata[myvar=='SEC0000431',]
emptylabels = mydata2$label==''
> length(emptylabels[emptylabels==FALSE]) 
[1] 158

> (67 / 30109)*100
[1] 0.2225248
> 

> emptylabels = mydata$label==''
> length(emptylabels[emptylabels==TRUE]) 
[1] 10745743
> length(emptylabels[emptylabels==FALSE]) 
[1] 48596
> 
> (48596 / 10745743)*100
[1] 0.4522349
> 


mydata2$binarylabels = !emptylabels
mydata$binarylabels = !emptylabels
binarylabels = mydata2$binarylabels

binarylabels <- gsub(FALSE, "N", binarylabels)
binarylabels <- gsub(TRUE, "P", binarylabels)
binarylabels = noquote(binarylabels)
mydata2$binarylabels = binarylabels

ggplot(mydata2, aes(x=timestampdiff, y=binarylabels)) +
  geom_point(size=5) +
  scale_x_continuous(breaks=c(10,20,30,40,50,60,70,80,90,100,110,120)) +
  scale_color_discrete("Alerts",labels=c("positive","negative")) +
  theme_bw()

export <- mydata2[,c("timestampdiff","price","binarylabels")]
write.csv(file="filteredsample.csv", x=export)

filtereddata = read.csv("filteredsample.csv")
export <- filtereddata[,c("timestampdiff","price","binarylabels")]
write.csv(file="filteredsample.csv", x=export)

"rowids","timestampdiff","price","binarylabels"

scp -r threemonth_sample_demomarket_1000seriesalerts.csv achivuku@titan9:/home/achivuku/Documents/BigLearningDatasets 
scp -r achivuku@titan9:/home/achivuku/Documents/BigLearningDatasets/filteredsample.csv /home/aneesh/Desktop/UTS\ Literature\ Survey/NASDAQ\ Project/

sed 's/\"//g' filteredsample.csv > input.csv

InFile metadata :
"timestampdiff","price","binarylabels"
date,time,instrument,transactionid,price,label

