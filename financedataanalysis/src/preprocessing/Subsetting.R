setwd("/home/achivuku/Documents/BigLearningDatasets")
mydata = read.csv("Alert1001_TradeOutput.csv")
Sys.setenv(TZ = "AEST")
options(digits.secs = 3);
temp = paste(mydata$Date, mydata$Time)[1]
strptime(temp, format="%d/%m/%Y %H:%M:%OS")
mydata$Timestamp = as.POSIXct(paste(mydata$Date, mydata$Time), format="%d/%m/%Y %H:%M:%OS")

emptylabels = is.na(mydata$alert1001)
binarylabels = !emptylabels
binarylabels <- gsub(FALSE, "N", binarylabels)
binarylabels <- gsub(TRUE, "P", binarylabels)
binarylabels = noquote(binarylabels)
mydata$binarylabels = binarylabels

length(unique(mydata$security))

for(v in unique(mydata$security)){
	print(c(v,nrow(mydata[mydata$security==v,])))
}

for(s in unique(mydata$security)){
    mydata2 = mydata[mydata$security==s,]
    startdate = as.character(mydata2[1,]$Timestamp)

    Timestampdiff <- 0
    for(i in 1:nrow(mydata2)) {
        Timestampdiff[i] <- difftime(mydata2[i,]$Timestamp,startdate,unit="secs")
    }
    mydata2$Timestampdiff = Timestampdiff

    export <- mydata2[,c("Timestampdiff","price","binarylabels")]
    write.csv(file=file.path("./inputs",paste(s,"filteredsample.csv",sep=".")), x=export,row.names=FALSE)
        
}

cd /home/achivuku/Documents/BigLearningDatasets/inputs
find ./ -exec sed -i 's/\"//g' {} \;

s = "SEC0000009"
mydata2 = mydata[mydata$security==s,]

emptylabels = is.na(mydata2$alert1001)
mydata3 = mydata2[emptylabels==FALSE,]
mydata3[emptylabels==FALSE,]$alert1001
length(unique(mydata3$security))

require(xts)
allpricexts <- xts(mydata2$price, order.by=mydata2$Timestamp, tz="AEST")
positivespricexts <- xts(mydata3$price, order.by=mydata3$Timestamp, tz="AEST")
i = 1
positivespricexts[i,]
time(positivespricexts)
time(positivespricexts[i,]) %in% time(positivespricexts)
d1 = time(positivespricexts[i,])
d2 = time(positivespricexts[i+3,])
positivespricexts[paste(as.character(d1),as.character(d1),sep="/")]
positivespricexts[paste(as.character(d1),as.character(d2),sep="/")]
positivespricexts[paste(as.character(d1 - 600),as.character(d2),sep="/")]
allpricexts[paste(as.character(d1 - 600),as.character(d1),sep="/")]

mydata2[which(mydata2$Timestamp < as.character(d1) & mydata2$Timestamp > as.character(d1 - 600)),]$price
# Proceed to the plotting logic using xts

startdate = as.character(mydata2[1,]$Timestamp)

Timestampdiff <- 0
for(i in 1:nrow(mydata2)) {
Timestampdiff[i] <- difftime(mydata2[i,]$Timestamp,startdate,unit="secs")
}

mydata2$Timestampdiff = Timestampdiff
head(mydata2)

export <- mydata2[,c("Timestampdiff","price","binarylabels")]
write.csv(file=file.path("./inputs",paste(s,"filteredsample.csv",sep=".")), x=export,row.names=FALSE)


cd /home/achivuku/Documents/BigLearningDatasets/inputs
find ./ -exec sed -i 's/\"//g' {} \;
sed 's/\"//g' filteredsample.csv > input.csv





deleting garbage rows with id 3963 and 2827 with many entries for 1001 : 2827,20121227,15:59:59.856,20121227 15:59:59.856,SEC0000094,BRK0000093,TRD0000182,CLI0025935,BRK0000119,???,CLI0003984,1110,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,1001,
do not distinguish between stocks to create positive and negative sample : load data only on server
using "SEC0000009" "588" for creating sample. repeat logic across securities to get final sample.
Implementing the preprocessing in R. But need to check alternatives in Python also. Start from following links.
xts like packages vignettes : zoo, quantmod, Rmetrics, its, 
Use python code with additional indexing attributes for creating data sample across securities and labels. Better to reformat data in expected format than do exploratory analysis in R. 
Have separate input file for each security generating alert 1001. Use same sampling/sequencing logic across each such file.

CRAN Task View: Time Series Analysis
https://cran.r-project.org/web/views/TimeSeries.html
Time Series / Date functionality
http://pandas.pydata.org/pandas-docs/stable/timeseries.html
TimeSeriesAnalysiswithPython
https://github.com/rouseguy/TimeSeriesAnalysiswithPython
What is the most useful Python library for time series and forecasting?
https://www.quora.com/What-is-the-most-useful-Python-library-for-time-series-and-forecasting
Great R packages for data import, wrangling and visualization
http://www.computerworld.com/article/2921176/business-intelligence/great-r-packages-for-data-import-wrangling-visualization.html
Time Series Analysis and Mining with R
https://www.slideshare.net/rdatamining/time-series-analysis-and-mining-with-r

