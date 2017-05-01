
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


myvar = mydata$label
table(myvar)

> table(myvar)
myvar
                            ???           |1001 
       10745743              23            3105 
     |1001|1005 |1001|1005|1040 |1001|1005|1050 
            777             113               1 
     |1001|1040      |1001|1050           |1005 
            229               2            5631 
     |1005|1020 |1005|1020|1040      |1005|1040 
           1071              84             727 
|1005|1040|1050      |1005|1050           |1020 
              2               2            9265 
     |1020|1040           |1040      |1040|1050 
             72            3111               1 
          |1050           |1060      |1060|1020 
          23727             594               3 
           1005       1005|1020            1020 
              3               3              46 
      1020|1005  1020|1005|1040 
              3               1 
> 

Class to record proportions :

1001, 1005, 1020, 1040, 1050, 1060

1001 : 3105 + 777 + 113 + 1 + 229 + 2 = 4227 4227 / 10745743 = 0.000393365
1005 : 777 + 113 + 1 + 5631 + 1071 + 84 + 727 + 2 + 2 + 3 + 3 + 3 + 1 = 8418  8418 / 10745743 = 0.00078338
1020 : 1071 + 84 + 9265 + 72 + 3 + 3 + 46 + 3 + 1 = 10548  10548 / 10745743 = 0.000981598
1040 : 113 + 229 + 84 + 727 + 2 + 72 + 3111 + 1 + 1 = 4340  4340 / 10745743 = 0.000403881
1050 : 1 + 2 + 2 + 2 + 1 + 23727 = 23735  23735 / 10745743 = 0.002208782
1060 : 594 + 3 = 597 597 / 10745743 = 0.000055557


discard milliseconds or avoid subsecond accuracy in visualizations

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

mydata[myvar=='SEC0000431',]
head(mydata[myvar=='SEC0000431',])
tail(mydata[myvar=='SEC0000431',])

mydata2 = mydata[myvar=='SEC0000271',]
mydata2 = mydata[myvar=='SEC0000431',]
myvar2 = mydata2$label

nrow(mydata2)
mydata2test = mydata2[1:100,]

timestamp = as.POSIXct(paste(mydata2test$date, mydata2test$time), format="%d/%m/%Y %H:%M:%OS")
timestamp = as.POSIXct(paste(mydata2$date, mydata2$time), format="%d/%m/%Y %H:%M:%OS")
mydata2$timestamp = timestamp
temp = "24/09/2012 23:37:00.000"

temp = paste(mydata2$date, mydata2$time)[1]
as.POSIXct(temp, format="%d/%m/%Y %H:%M:%OS3")
format(as.POSIXct(temp,format="%d/%m/%Y %H:%M:%OS"), "%d/%m/%Y %H:%M:%OS3")
format.POSIXlt(temp, "%d/%m/%Y %H:%M:%OS3")
strptime(temp, format="%d/%m/%Y %H:%M:%OS")

Sys.setenv(TZ = "AEST")
options(digits.secs = 3);
strptime(temp, format="%d/%m/%Y %H:%M:%OS")

discard milliseconds
POSIXct subsecond output [duplicate]
http://stackoverflow.com/questions/17571615/posixct-subsecond-output
How R formats POSIXct with fractional seconds
http://stackoverflow.com/questions/7726034/how-r-formats-posixct-with-fractional-seconds
Date-time Conversion Functions to and from Character
https://stat.ethz.ch/R-manual/R-devel/library/base/html/strptime.html
Date-time Conversion Functions
https://stat.ethz.ch/R-manual/R-devel/library/base/html/as.POSIXlt.html



difftime("2012-09-03 09:30:06", "2012-09-03 09:30:01 AEST", unit="weeks")
difftime("2012-09-03 09:30:06", "2012-09-03 09:30:01 AEST", unit="secs")

difftime(mydata2[2,]$timestamp,mydata2[1,]$timestamp)
difftime(mydata2[2,]$timestamp,"2012-09-03 09:30:01 AEST")
mydata2[1,]$timestamp : "2012-09-03 09:30:01 AEST"

timestampdiff <- 0
for(i in 1:nrow(mydata2)) {
timestampdiff[i] <- difftime(mydata2[i,]$timestamp,"2012-09-03 09:30:01.001 AEST",unit="secs")
}

timestampdiff <- 0
for(i in 1:nrow(mydata2)) {
timestampdiff[i] <- difftime(mydata2test[i,]$timestamp,"2012-09-03 09:30:01.001 AEST",unit="secs")
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


> mydata2[which(mydata2$label=='|1001|1005|1040'),]
              date         time instrument transactionid price
2035054 24/09/2012 13:54:01.588 SEC0000431     167137533   510
                  label           timestamp timestampdiff
2035054 |1001|1005|1040 2012-09-24 13:54:01       1830241

> mydata2[which(mydata2$label=='|1020|1040'),]
             date         time instrument transactionid price
423027 07/09/2012 09:30:00.202 SEC0000431     165525506   455
            label           timestamp timestampdiff
423027 |1020|1040 2012-09-07 09:30:00      345599.2
> 

> mydata2[which(mydata2$label=='|1005|1040'),]
              date         time instrument transactionid price
371456  06/09/2012 13:47:10.426 SEC0000431     165473935   445
374745  06/09/2012 13:59:34.534 SEC0000431     165477224   450
451113  07/09/2012 10:08:53.377 SEC0000431     165553592   470
1200172 14/09/2012 14:56:40.473 SEC0000431     166302651   480
2029936 24/09/2012 13:37:51.672 SEC0000431     167132415   490
2104857 25/09/2012 09:41:32.518 SEC0000431     167207336   520
3515777 09/10/2012 10:59:36.586 SEC0000431     168618256   580
3572858 09/10/2012 15:20:59.590 SEC0000431     168675337   590
3654199 10/10/2012 10:43:23.416 SEC0000431     168756678   610
             label           timestamp timestampdiff
371456  |1005|1040 2012-09-06 13:47:10      274629.4
374745  |1005|1040 2012-09-06 13:59:34      275373.5
451113  |1005|1040 2012-09-07 10:08:53      347932.4
1200172 |1005|1040 2012-09-14 14:56:40      969999.5
2029936 |1005|1040 2012-09-24 13:37:51     1829270.7
2104857 |1005|1040 2012-09-25 09:41:32     1901491.5
3515777 |1005|1040 2012-10-09 10:59:36     3112175.6
3572858 |1005|1040 2012-10-09 15:20:59     3127858.6
3654199 |1005|1040 2012-10-10 10:43:23     3197602.4
>

mydata2[which(mydata2$label=='|1005|1040'),]



> mydata2[which(mydata2$timestampdiff < 345600 & mydata2$timestampdiff > 345500),]
             date         time instrument transactionid price
423027 07/09/2012 09:30:00.202 SEC0000431     165525506   455
423028 07/09/2012 09:30:00.215 SEC0000431     165525507   455
423029 07/09/2012 09:30:00.220 SEC0000431     165525508   455
            label           timestamp timestampdiff
423027 |1020|1040 2012-09-07 09:30:00      345599.2
423028            2012-09-07 09:30:00      345599.2
423029            2012-09-07 09:30:00      345599.2
> 

> nrow(mydata2[which(mydata2$timestampdiff < 345700 & mydata2$timestampdiff > 345100),])
[1] 160

subset for +/-10 minutes from alert time stamp

> nrow(mydata2[which(mydata2$timestampdiff < 345900 & mydata2$timestampdiff > 345100),])
[1] 477

> nrow(mydata2[which(mydata2$timestampdiff < 346000 & mydata2$timestampdiff > 340000),])
[1] 547

nrow(mydata2[which(mydata2$timestampdiff < 1830541 & mydata2$timestampdiff > 1829941),])

nrow(mydata2[which(mydata2$timestampdiff < 1901792 & mydata2$timestampdiff > 1901191),])


> emptylabels = mydata2[which(mydata2$timestampdiff < 1901792 & mydata2$timestampdiff > 1901191),]$label=='' 
> length(emptylabels[emptylabels==FALSE])
[1] 1

> emptylabels = mydata2[which(mydata2$timestampdiff < 345700 & mydata2$timestampdiff > 345100),]$label=='' 
> length(emptylabels[emptylabels==FALSE])
[1] 2

> emptylabels = mydata2[which(mydata2$timestampdiff < 1830541 & mydata2$timestampdiff > 1829941),]$label=='' 
> length(emptylabels[emptylabels==FALSE])
[1] 2

emptylabels = mydata2$label==''
mydata2[emptylabels==FALSE,]$label

mydata3 = mydata2[which(mydata2$timestampdiff < 1901792 & mydata2$timestampdiff > 1901191),]
> unique(mydata3$instrument)
[1] SEC0000431
495 Levels: SEC0000001 SEC0000002 SEC0000003 ... SEC0003286
>

mydata3 = mydata2[which(mydata2$timestampdiff < 345700 & mydata2$timestampdiff > 345100),]
> unique(mydata3$instrument)
[1] SEC0000431
495 Levels: SEC0000001 SEC0000002 SEC0000003 ... SEC0003286
> 

mydata3 = mydata2[which(mydata2$timestampdiff < 1830541 & mydata2$timestampdiff > 1829941),]
> unique(mydata3$instrument)
[1] SEC0000431
495 Levels: SEC0000001 SEC0000002 SEC0000003 ... SEC0003286
> 

Also subset data by SEC0000431 and show time series across all time stamps in alerts

dygraph(pricexts, main = "Instrument Prices") %>% 
  dyHighlight(highlightSeriesOpts = list(strokeWidth = 3))

dygraph(series, main = "Instrument Prices") %>% 
    dyHighlight(highlightSeriesOpts = list(strokeWidth = 3)) %>% 
    dyEvent("2015-01-01 17:00:00", "|1005|1040", labelLoc = "bottom", color="blue")

Sys.setenv(TZ = "AEST")

length(mydata3$timestamp) == length(unique(mydata3$timestamp))
To use annotation in dygraphs, use ts object instead of xts object
Cannot use ts object. Compare with series xts example in docs. Make timestamps unique.
To have unique timestamp for records, do not discard fractional seconds when creating timestamp.
Even with full timestamp, records have duplicate timestamps
Also fractional seconds do not show in dygraphs output
Event lines cannot show with non-unique time stamps
Checking alternatives to dygraphs for plotting xts data

https://www.r-bloggers.com/plot-xts-is-wonderful/
https://gist.github.com/timelyportfolio/3373828
http://blog.revolutionanalytics.com/2014/01/quantitative-finance-applications-in-r-plotting-xts-time-series.html

pricexts <- xts(mydata3$price, order.by=mydata3$timestamp, tz="AEST")
ts(log(pricexts), start = start(pricexts), end = end(pricexts))


pricextstest = pricexts[1:10,]



dygraph(pricexts)

dygraph(pricextstest)  %>% 
  dyOptions(useDataTimezone = TRUE, digitsAfterDecimal=3)  %>% 
  dyEvent("2012-09-25 09:36:00.000", "|1005|1040", labelLoc = "bottom", color="blue")


dygraph(pricexts, main = "Instrument Prices") %>% 
    dyHighlight(highlightSeriesOpts = list(strokeWidth = 3)) %>% 
    dyEvent("2012-09-25 09:38:00", "|1005|1040", labelLoc = "bottom", color="blue")

dygraph(pricexts, main = "Instrument Prices", tz="AEST") %>% 
  dyAxis("y", label = "Prices", valueRange = c(495, 525)) %>% 
  dyOptions(drawPoints = TRUE, pointSize = 2) %>% 
  dyEvent("2012-09-25 09:41:32", "|1005|1040", labelLoc = "bottom", color="blue")

dygraph(pricexts, main = "Instrument Prices") %>% 
  dyAxis("y", label = "Prices", valueRange = c(495, 525)) %>% 
  dyOptions(drawPoints = TRUE, pointSize = 2) %>% 
  dyEvent("2012-09-25 09:41:32", "|1005|1040") %>% 
  dyAnnotation("2012-09-25 09:41:32", text = "A", tooltip = "|1005|1040")

dygraph(pricexts, main = "Instrument Prices") %>% 
  dyAxis("y", label = "Prices", valueRange = c(495, 525)) %>% 
  dyAnnotation("2012-09-25 09:41:32", text = "A", tooltip = "|1005|1040")

dygraph(pricexts, main = "Instrument Prices") %>% 
  dyAxis("y", label = "Prices", valueRange = c(495, 525)) %>% 
  dyShading(from = "2012-09-25 09:40:32", to = "2012-09-25 09:41:32")

  dyAnnotation("2012-09-25 09:41:32", text = "A", tooltip = "|1005|1040") %>% 


dygraph(presidents, main = "Quarterly Presidential Approval Ratings") %>%
  dyAxis("y", valueRange = c(0, 100)) %>%
  dyEvent("1950-6-30", "Korea", labelLoc = "bottom") %>%
  dyEvent("1965-2-09", "Vietnam", labelLoc = "bottom")

Check parameters of plot.xts, ggplot2, plot.xts, 

require(xtsExtra)
xts::plot.xts(pricexts)
https://gist.github.com/timelyportfolio/3373828
https://www.r-bloggers.com/plot-xts-is-wonderful/

xts, quantmod, ggplot2
http://stackoverflow.com/questions/35215579/how-to-plot-xts-in-ggplot2
http://joshuaulrich.github.io/xts/plotting_basics.html


zoo.pricextstest <- as.zoo(pricextstest)
tsRainbow <- rainbow(ncol(zoo.pricextstest))
plot(x = zoo.pricextstest, col = tsRainbow, screens = 1)
myColors <- c("red", "darkgreen", "goldenrod", "darkblue", "darkviolet")
plot(x = zoo.pricextstest, xlab = "Time", ylab = "Price", main = "Instrument Price", col = myColors, screens = 1)

legend(x = "topleft", legend = c("SPY", "QQQ", "GDX", "DBO", "VWO"),
       lty = 1, col = myColors)

xts::plot.xts(x = pricextstest, xlab = "Time", ylab = "Price",
main = "Instrument Price", ylim = c(495, 525), major.ticks= "minutes",
        minor.ticks = FALSE, col = "red")



xts::plot.xts(x = pricexts, xlab = "Time", ylab = "Price",
main = "Instrument Price", ylim = c(495, 525), major.ticks= "minutes",
        minor.ticks = FALSE, col = "blue")

xts::plot.xts(x = pricexts, main = "Price vs Time", ylim = c(495, 525), major.ticks= "minutes", minor.ticks = FALSE, col = "blue")
xts::plot.xts(x = pricexts, main = "Price vs Time", ylim = c(440, 465), major.ticks= "minutes", minor.ticks = FALSE, col = "blue")
xts::plot.xts(x = pricexts, main = "Price vs Time", ylim = c(485, 515), major.ticks= "minutes", minor.ticks = FALSE, col = "blue")
xts::plot.xts(x = pricexts, main = "Price vs Time", ylim = c(305, 645), major.ticks= "days", minor.ticks = FALSE, col = "blue")

library(xts)

emptylabels = mydata3$label=='' 
mydata3[emptylabels==FALSE,]

anomaly.dates = c("24/09/2012 23:41:32.517")
anomaly.labels = c("1005 and 1040")
anomaly.instrument = c("SEC0000431")

anomaly.dates = c("24/09/2012 03:52:25.595","24/09/2012 03:54:01.588")
anomaly.labels = c("1040","1001, 1005 and 1040")
anomaly.instrument = c("SEC0000431")






2012-09-25 09:41:32



addEventLines(anomaly.dates, anomaly.labels)
addEventLines(event.dates = anomaly.dates, event.labels = anomaly.labels, date.format = "%d/%m/%Y %H:%M:%OS",on=1,offset=.4, pos=2, srt=90, cex=1.5, col="red",font=20)

legend(x = 'topleft', legend = c("SPY", "QQQ", "GDX", "DBO", "VWO"),
      lty = 1, col = myColors)


Need to find a package that gives plots and event lines














setwd("/home/achivuku/Documents/BigLearningDatasets")
mydata = read.csv("threemonth_sample_demomarket_1000seriesalerts.csv")
myvar = mydata$instrument
mydata2 = mydata[myvar=='SEC0000431',]

Sys.setenv(TZ = "AEST")
options(digits.secs = 3);

timestamp = as.POSIXct(paste(mydata2$date, mydata2$time), format="%d/%m/%Y %H:%M:%OS")
mydata2$timestamp = timestamp

timestampdiff <- 0
for(i in 1:nrow(mydata2)) {
timestampdiff[i] <- difftime(mydata2[i,]$timestamp,"2012-09-03 09:30:01.001 AEST",unit="secs")
}

mydata2$timestampdiff = timestampdiff

mydata2[which(mydata2$label=='|1001|1005|1040'),]
mydata2[which(mydata2$label=='|1020|1040'),]
mydata2[which(mydata2$label=='|1005|1040'),] - 

mydata2[which(mydata2$timestampdiff < 345600 & mydata2$timestampdiff > 345500),]

emptylabels = mydata2[which(mydata2$timestampdiff < 1901792 & mydata2$timestampdiff > 1901191),]$label==''
length(emptylabels[emptylabels==FALSE])

emptylabels = mydata2[which(mydata2$timestampdiff < 345700 & mydata2$timestampdiff > 345100),]$label=='' 
length(emptylabels[emptylabels==FALSE])

emptylabels = mydata2[which(mydata2$timestampdiff < 1830541 & mydata2$timestampdiff > 1829941),]$label=='' 
length(emptylabels[emptylabels==FALSE])





emptylabels = mydata2[which(mydata2$timestampdiff < 1831441 & mydata2$timestampdiff > 1829041),]$label=='' 
length(emptylabels[emptylabels==FALSE])

mydata3 = mydata2[which(mydata2$timestampdiff < 1901792 & mydata2$timestampdiff > 1901191),]
mydata3 = mydata2[which(mydata2$timestampdiff < 1902691 & mydata2$timestampdiff > 1900291),]

mydata3 = mydata2[which(mydata2$timestampdiff < 345700 & mydata2$timestampdiff > 345100),]

mydata3 = mydata2[which(mydata2$timestampdiff < 1830541 & mydata2$timestampdiff > 1829941),]
mydata3 = mydata2[which(mydata2$timestampdiff < 1831441 & mydata2$timestampdiff > 1829041),]

emptylabels = mydata3$label==''
mydata3[emptylabels==FALSE,]$label
mydata3[emptylabels==FALSE,]



unique(mydata3$instrument)

length(mydata3$timestamp) == length(unique(mydata3$timestamp))

pricexts <- xts(mydata3$price, order.by=mydata3$timestamp, tz="AEST")

library(xts)
xts::plot.xts(x = pricexts, main = "Price vs Time", ylim = c(495, 525), major.ticks= "minutes", minor.ticks = FALSE, col = "blue")
xts::plot.xts(x = pricexts, main = "Price vs Time", ylim = c(440, 465), major.ticks= "minutes", minor.ticks = FALSE, col = "blue")
xts::plot.xts(x = pricexts, main = "Price vs Time", ylim = c(485, 515), major.ticks= "minutes", minor.ticks = FALSE, col = "blue")
xts::plot.xts(x = pricexts, main = "Price vs Time", ylim = c(305, 645), major.ticks= "days", minor.ticks = FALSE, col = "blue")
addEventLines(event.dates = anomaly.dates, event.labels = anomaly.labels, date.format = "%d/%m/%Y %H:%M:%OS",on=1,offset=.4, pos=2, srt=90, cex=1.5, col="red",font=20)


emptylabels = mydata3$label=='' 
mydata3[emptylabels==FALSE,]

anomaly.dates = c("24/09/2012 23:41:32.517")
anomaly.labels = c("1005 and 1040")
anomaly.instrument = c("SEC0000431")

anomaly.dates = c("24/09/2012 03:52:25.595","24/09/2012 03:54:01.588")
anomaly.labels = c("1040","1001, 1005 and 1040")
anomaly.instrument = c("SEC0000431")








anomaly.dates = c("24/09/2012 13:37:51.671","24/09/2012 13:44:27.823","24/09/2012 13:52:25.595","24/09/2012 13:54:01.588")
anomaly.labels = c("1005, 1040","1040","1040","1001, 1005 and 1040")
anomaly.instrument = c("SEC0000431")

xts::plot.xts(x = pricexts, main = "Price vs Time", ylim = c(460, 515), major.ticks= "days", minor.ticks = FALSE, col = "blue")
addEventLines(event.dates = anomaly.dates, event.labels = anomaly.labels, date.format = "%d/%m/%Y %H:%M:%OS",on=1,offset=.4, pos=2, srt=90, cex=1.5, col="red",font=20)

anomaly.dates = c("24/09/2012 13:37:51.671","24/09/2012 13:44:27.823","24/09/2012 13:52:25.595","24/09/2012 13:54:01.588")
anomaly.labels = c("1005, 1040","1040","1040","1001, 1005 and 1040")
anomaly.instrument = c("SEC0000431")

xts::plot.xts(x = pricexts, main = "Price vs Time", ylim = c(460, 515), major.ticks= "days", minor.ticks = FALSE, col = "blue")
addEventLines(event.dates = anomaly.dates, event.labels = anomaly.labels, date.format = "%d/%m/%Y %H:%M:%OS",on=1,offset=.4, pos=2, srt=90, cex=1.5, col="red",font=20)

anomaly.dates = c("25/09/2012 09:30:00.381","25/09/2012 09:41:32.518")
anomaly.labels = c("1005","1005 and 1040")
anomaly.instrument = c("SEC0000431")

xts::plot.xts(x = pricexts, main = "Price vs Time", ylim = c(490, 525), major.ticks= "days", minor.ticks = FALSE, col = "blue")
addEventLines(event.dates = anomaly.dates, event.labels = anomaly.labels, date.format = "%d/%m/%Y %H:%M:%OS",on=1,offset=.4, pos=2, srt=90, cex=1.5, col="red",font=20)



> mydata2[which(mydata2$label=='|1001|1005|1040'),]
              date         time instrument transactionid price
2035054 24/09/2012 13:54:01.588 SEC0000431     167137533   510
                  label               timestamp timestampdiff
2035054 |1001|1005|1040 2012-09-24 13:54:01.588       1830241
> mydata2[which(mydata2$label=='|1020|1040'),]
             date         time instrument transactionid price
423027 07/09/2012 09:30:00.202 SEC0000431     165525506   455
            label               timestamp timestampdiff
423027 |1020|1040 2012-09-07 09:30:00.201      345599.2
> mydata2[which(mydata2$label=='|1005|1040'),]
              date         time instrument transactionid price
371456  06/09/2012 13:47:10.426 SEC0000431     165473935   445
374745  06/09/2012 13:59:34.534 SEC0000431     165477224   450
451113  07/09/2012 10:08:53.377 SEC0000431     165553592   470
1200172 14/09/2012 14:56:40.473 SEC0000431     166302651   480
2029936 24/09/2012 13:37:51.672 SEC0000431     167132415   490
2104857 25/09/2012 09:41:32.518 SEC0000431     167207336   520
3515777 09/10/2012 10:59:36.586 SEC0000431     168618256   580
3572858 09/10/2012 15:20:59.590 SEC0000431     168675337   590
3654199 10/10/2012 10:43:23.416 SEC0000431     168756678   610
             label               timestamp timestampdiff
371456  |1005|1040 2012-09-06 13:47:10.426      274629.4
374745  |1005|1040 2012-09-06 13:59:34.533      275373.5
451113  |1005|1040 2012-09-07 10:08:53.377      347932.4
1200172 |1005|1040 2012-09-14 14:56:40.473      969999.5
2029936 |1005|1040 2012-09-24 13:37:51.671     1829270.7
2104857 |1005|1040 2012-09-25 09:41:32.517     1901491.5
3515777 |1005|1040 2012-10-09 10:59:36.585     3115775.6
3572858 |1005|1040 2012-10-09 15:20:59.589     3131458.6
3654199 |1005|1040 2012-10-10 10:43:23.415     3201202.4


















setwd("/home/achivuku/Documents/BigLearningDatasets")
mydatao = read.csv("Alert1001_TradeOutput.csv")
head(mydatao)
l = length(unique(mydatao$security))
print(l)
df <- data.frame(security = character(520), numpositives = numeric(520), positivesmean = numeric(520), positivesvariance = numeric(520), stringsAsFactors = FALSE)
i = 1
for(v in unique(mydatao$security)){
    mydataos = mydatao[mydatao$security==v,]
    emptylabels = is.na(mydataos$alert1001)
    np = nrow(mydataos[emptylabels==FALSE,])
    mn = mean(mydataos[emptylabels==FALSE,]$price)
    vn = var(mydataos[emptylabels==FALSE,]$price)
	print(c(v,np,mn,vn))
    df$security[i] <- v
    df$numpositives[i] <- np
    df$positivesmean[i] <- mn
    df$positivesvariance[i] <- vn
}
print(df)
write.csv(file="summarystats.csv", x=df)

