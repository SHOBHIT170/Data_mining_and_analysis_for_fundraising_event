## MSIS 672 - Final Project
## Sushobhit Nandkishor Lutade

# Read the data and create a data frame for the data set 
fund.df <- read.csv("Fundraising.csv")
str(fund.df)
summary(fund.df)

# select input variables for TARGET B (eliminate id columns and target D)
targetB.df <- fund.df[, -c(1,2,24)]

### TARGET B Classification Tree method
# Data partition
# Split the data into training (60%) and validation (40%)
set.seed(12345)
train.index <- sample(dim(targetB.df)[1], dim(targetB.df)[1]*0.6)
trainB.df <- targetB.df[train.index,]
validB.df <- targetB.df[-train.index,]


library(rpart)
library(rpart.plot)
library(caret)

# Create a classification tree  (TARGET B is the target variable)
# Display cp table to choose the best cp value
cv.ct <- rpart(TARGET_B ~ ., data = trainB.df, method = "class", cp = 0.0001, minsplit = 1, xval = 5)  # minsplit is the minimum number of observations in a node for a split to be attempted. xval is number K of folds in a K-fold cross-validation.
printcp(cv.ct) 

# Create a classification tree  (TARGET B is the target variable, with choosen cp value from above)
TARGET_B.ct <- rpart(TARGET_B ~ ., data = trainB.df, method = "class", cp = 0.008306, minsplit = 1)
# Display model diagram
prp(TARGET_B.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10)
# Display model rules 
rpart.rules(TARGET_B.ct)


# Use the model to predict Competitive on training data
TARGET_B.ct.point.pred.train <- predict(TARGET_B.ct,trainB.df,type = "class")
# model's performance on the training data
confusionMatrix(TARGET_B.ct.point.pred.train, as.factor(trainB.df$TARGET_B),positive = "1")
# Use the model to predict Competitive on validation data
TARGET_B.ct.point.pred.valid <- predict(TARGET_B.ct,validB.df,type = "class")
# model's performance on the validation data 
confusionMatrix(TARGET_B.ct.point.pred.valid, as.factor(validB.df$TARGET_B),positive = "1")


### TARGET B Neural Network method
library(neuralnet)
library(nnet)
library(caret)
library(e1071)

# Selected important input variable from classification tree
inputVars <- c("MAXRAMNT", "totalmonths", "NUMCHLD", "Icavg", "NUMPROM", "LASTGIFT", "INCOME", "Icmed", "IC15")
# Save as neural network input variables 
inputVars.nn<-targetB.df[,inputVars]

# normalize_min_maz function
# calcuate normlized values using miminum-maximum formula and then return normalized values
normalize_min_maz<- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Normalize all neural network input variables 
inputVars.nn <- normalize_min_maz(inputVars.nn)

# Combine input and output variables for nn model
targetB.nn<-cbind(inputVars.nn, targetB.df$TARGET_B)
names(targetB.nn)[10] <- 'TARGET_B'

# Split the data into training (60%) and validation (40%) using training index above
trainBnn.df <- targetB.nn[train.index,]
validBnn.df <- targetB.nn[-train.index,]

# run nn with 2 hidden nodes 
nn <- neuralnet(TARGET_B ~ MAXRAMNT + totalmonths + NUMCHLD + Icavg + NUMPROM + LASTGIFT + INCOME + Icmed + IC15, data = trainBnn.df, hidden = 2)
plot (nn, rep="best")

## Apply neuralnet model on training data 
# Create prediction weights with nn model using only input variables
training.prediction <- predict(nn, trainBnn.df) 
# Round the results with threshold of 0.5
roundedresults<-sapply(training.prediction,round,digits=0)
# factor the data for outputs
ytrain<-as.factor(trainBnn.df$TARGET_B)
# Create confusionMatrix for training data
confusionMatrix(as.factor(roundedresults),ytrain,positive = "1")

## Apply similar approach on validation data 
validation.prediction <- predict(nn, validBnn.df) 
roundedresults<-sapply(validation.prediction,round,digits=0)
yvalid<-as.factor(validBnn.df$TARGET_B)
confusionMatrix(as.factor(roundedresults),yvalid,positive = "1")


### TARGET D Linear Regression
targetD.df <- fund.df[, -c(1,2)]
targetD.df <- targetD.df[which(targetD.df$TARGET_B==1), ]

targetD.df <- targetD.df[, -c(21)]
scaledTargetD.df <- as.data.frame(scale(targetD.df, center=TRUE, scale=TRUE))
View(targetD.df)
View(scaledTargetD.df)
summary(targetD.df)
summary(scaledTargetD.df)
trainD.df=targetD.df[trainB.index,]
validD.df=targetD.df[-trainB.index,]
scaledTrainD.df=scaledTargetD.df[trainB.index,]
scaledValidD.df=scaledTargetD.df[-trainB.index,]


targetD.lm <- lm(TARGET_D ~ ., data = trainD.df)
scaledTragetD.lm <- lm(TARGET_D ~ ., data = scaledTrainD.df)
targetD.lm
options(scipen = 999, digits = 0)
summary(targetD.lm)
summary(scaledTragetD.lm)


library(forecast)
targetD.lm.pred <- predict(targetD.lm, validD.df)
scaledTargetD.lm.pred <- predict(scaledTragetD.lm, validD.df)
options(scipen=999, digits = 3)
accuracy(targetD.lm.pred, validD.df$TARGET_D)
accuracy(scaledTargetD.lm.pred, scaledValidD.df$TARGET_D)

##Compare 3 models, select the best one
#BACKWARD
targetDbw.lm.step <- step(targetD.lm, direction = "backward")
summary(targetDbw.lm.step)  # Which variables were dropped?
targetDbw.lm.step.pred <- predict(targetDbw.lm.step, validD.df)
accuracy(targetDbw.lm.step.pred, validD.df$TARGET_D)

#FORWARD
targetD.lm.null <- lm(TARGET_D~1, data = validD.df)
summary (targetD.lm.null)
targetDfw.lm.step <- step(targetD.lm.null, scope=list(lower=targetD.lm.null, upper=targetD.lm), direction = "forward")
summary(targetDfw.lm.step)  # Which variables were added?
targetDfw.lm.step.pred <- predict(targetDfw.lm.step, validD.df)
accuracy(targetDfw.lm.step.pred, validD.df$TARGET_D)

#STEPWISE
targetD.lm.step <- step(targetD.lm, direction = "both")
summary(targetD.lm.step)  # Which variables were dropped/added?
targetD.lm.step.pred <- predict(targetD.lm.step, validD.df)
accuracy(targetD.lm.step.pred, validD.df$TARGET_D)


