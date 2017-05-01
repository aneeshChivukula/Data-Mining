

setwd("/home/achivuku/Documents/BigLearningDatasets")
mydata = read.csv('sample1.csv',header=FALSE)
drops <- c("V601")
head(mydata[,!(names(mydata) %in% drops)])
mymatrix = data.matrix(mydata[,!(names(mydata) %in% drops)])

mymatrixmean = apply(mymatrix,1,mean)
mymatrixvar = apply(mymatrix,1,var)

mymatrix2 = (mymatrix - mymatrix[,1])
mymatrix3 = (mymatrix2 / mymatrix[,1])
head(mymatrix3,1)[2]

head(((mymatrix - mymatrix[,1]) / mymatrix[,1]),1)[2]

mymatrix2 = ((mymatrix - mymatrix[,1]) / mymatrix[,1])

mymatrix2mean = apply(mymatrix2,1,mean)
mymatrix2var = apply(mymatrix2,1,var)

class = mydata$V601
mymatrixstatsdf = cbind.data.frame(mymatrix2,mymatrixmean,mymatrixvar,mymatrix2mean,mymatrix2var,class)
head(mymatrixstatsdf)

smp_size <- floor(0.75 * nrow(mymatrixstatsdf))
set.seed(123)
train_ind <- sample(seq_len(nrow(mymatrixstatsdf)), size = smp_size)
training <- mymatrixstatsdf[train_ind, ]
testing <- mymatrixstatsdf[-train_ind, ]

library(e1071)
model <- naiveBayes(class ~ ., data = training)

class(model)
summary(model)
print(model)

pred_nb <- predict(model, newdata = testing)
truth <- testing$class

pred <- pred_nb

xtab <- table(pred, truth)
library(caret) 
confusionMatrix(xtab)

library(caret) 
set.seed(400)
ctrl <- trainControl(method="repeatedcv",repeats = 3) #,classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(class ~ ., data = train, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)
knnFit

pred_knn <- predict(knnFit,newdata = test )
pred <- pred_knn

xtab <- table(pred, truth)
confusionMatrix(xtab)





import numpy as np
from pandas import DataFrame
df = DataFrame.from_csv('sample.csv', sep=',')
X = df.values
X[0:10]

X_train = (X[:,0:-1])
y_train = (X[:,-1])
y_train = y_train.reshape(101, 1)
y_train[y_train == 'P'] = 1
y_train[y_train == 'N'] = 0
y_train = y_train.astype(np.int)

X_train.shape[1:]
Y_train.shape

X_train.shape[1:]

nb_classes = 2
batch_size = 101
nb_epoch = 20

Y_train = np_utils.to_categorical(y_train, nb_classes)

X_test = X_train
Y_test = Y_train

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
model = Sequential()

model.add(Dense(512, input_shape=(3599,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2))

model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.predict(X_test)

Examples for Multilayer Perceptron in Keras :

mnist_mlp.py
mnist_sklearn_wrapper.py
antirectifier.py
cifar10_cnn.py

Check docs and examples in keras-master on github webpage


