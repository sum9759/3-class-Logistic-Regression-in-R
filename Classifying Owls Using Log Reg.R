## Load the required libraries and if required, installation command is provided 
## for basic R libraries

#install.packages("readr")
#install.packages("dplyr")
#install.packages("ggplot2")
library(readr)
library(dplyr)
library(magrittr)
library(ggplot2)


## Loading the dataset
owls_data <- read_csv("Load the file")

## Naming the columns of the dataset
colnames(owls_data) <- c("body_length","wing_length", "body_width", "wing_width", "type")

## Visualizing the data given to identify the relationships amongst the variables
ggplot(owls_data, aes(x = body_length, y = wing_length))+ geom_point(aes(colour=type, shape=type), size = 3,position = "jitter")+ xlab("Body length")+ 
  ylab("Wing length")+ ggtitle("Wing_Length vs Body_Length of an owl")

ggplot(owls_data, aes(x = body_length, y = body_width))+ geom_point(aes(colour=type, shape=type), size = 3,position = "jitter")+ xlab("Body length")+ 
  ylab("Wing length")+ ggtitle("Body_Width vs Body_Length")

ggplot(owls_data, aes(x = body_length, y = wing_width))+ geom_point(aes(colour=type, shape=type), size = 3,position = "jitter")+ xlab("Body length")+ 
  ylab("Wing length")+ ggtitle("Wing_Width vs Body_Length")


### defining the functions for implementing a logistic classifier
## Here, I have used the one vs all algorithm to classify the 3 classes of owls

#sigmoid function, inverse of logit
sigmoid <- function(x){
  1/(1+exp(-x))
  }

#cost function 
## %*% is the dot product in R.
cost <- function(theta, X, Y){
  m <- length(Y) # number of training examples
  
  h <- sigmoid(X %*% theta)
  J <- (t(-Y)%*%log(h)-t(1-Y)%*%log(1-h))/m
  J
}


#gradient function
## Here, I am using gradient descent to converge my cost
gradient <- function(theta, X, Y){
  m <- length(Y) 
  
  h <- sigmoid(X %*% theta)
  gradient_descent <- (t(X)%*%(h - Y))/m
  gradient_descent
}


### Defining my logistic regression fuction, 
##  which takes owls_tr dataframe X (training dataset), 
##  and class Y as function input.
##  Its job is to return a column vector which stores the coefficients in theta. 
##  Note: Since the input X here does not have a bias term, we add it manually.
logisticReg <- function(X, Y){
  #remove NA rows
  temp <- na.omit(cbind(Y, X))
  #add bias term and convert to matrix
  X <- mutate(temp[, -1], bias =1)
  X <- as.matrix(X[,c(ncol(X), 1:(ncol(X)-1))])
  y <- as.matrix(temp[, 1])
  #initialize theta
  theta <- matrix(rep(0, ncol(X)), nrow = ncol(X))
  #use the optim function to perform gradient descent
  costOpti <- optim(matrix(rep(0, 5), nrow = 5), cost, gradient, X=X, Y=Y)
  #return coefficients
  return(costOpti$par)
}



## defining the logistic prediction function that returns the probablity of seeing a 
## class provided a given test case
logisticPredictor <- function(mod, X){
  X <- na.omit(X)
  #add bias term and convert to matrix
  X <- mutate(X, bias = 1)
  X <- as.matrix(X[,c(ncol(X), 1:(ncol(X)-1))])
  return(sigmoidx(X %*% mod))
}

##### Making 10 different samples using for loop to test individual accuracy and 
## average accuracy of the defined logistic predictor for the owls dataset

# Initializing some variables
predicted = accuracy <- 0

## Running 10 iterations 
for(a in 1:10){
  ## randomly sampling data without replacement
index <- sample(1: nrow(owls_data),94,replace = FALSE) 
## dividing the data into train and test sets
owls_tr <- owls_data[index,]
owls_test1 <- owls_data[-index,]
owls_test <- owls_test1[,c(-5)]

## Here, we have implemented one vs all algorithm to predict multiclass using our logistic predictor
## making BarnOwls and SnowyOwls class of owls 0 and LongEaredOwl to 1
owls_type1 <- owls_tr
owls_type1$type <- ifelse(owls_type1$type == "LongEaredOwl", 1,0)

## making LongEared and SnowyOwls class to 0, BarnOwl to 1
owls_type2 <- owls_tr
owls_type2$type <- ifelse(owls_type2$type == "BarnOwl", 1,0)

## making LongEared and Barn Owl to 0, SnowyOwl to 1
owls_type3 <- owls_tr
owls_type3$type <- ifelse(owls_type3$type == "SnowyOwl", 1,0)


## Forming three separate datasets for training and testing for 3 classes of owls
owls.X1 <- owls_type1[, -5]
owls.y1 <- owls_type1[, 5]

owls.X2 <- owls_type2[, -5]
owls.y2 <- owls_type2[, 5]

owls.X3 <- owls_type3[, -5]
owls.y3 <- owls_type3[, 5]

## Training our model on the data for the 3 classes
mod1 <- logisticReg(owls.X1, owls.y1)
mod2 <- logisticReg(owls.X2, owls.y2)
mod3 <- logisticReg(owls.X3, owls.y3)

## For all the test cases, we predict the class as the one which is max among the 3 predictors probabalities
for(i in 1:nrow(owls_test)){
  
FG1 <- logisticPredictor(mod1,owls_test[i,])
FG2 <- logisticPredictor(mod2,owls_test[i,])
FG3 <- logisticPredictor(mod3,owls_test[i,])
predicted[i] <- ifelse(FG1>FG2 & FG1>FG3, "LongEaredOwl", ifelse(FG2>FG1 & FG2>FG3, "BarnOwl","SnowyOwl"))
#print("Class of test case")
#print(owls_test[i,])
#print(predicted[i])

}
## creating a dataframe of actual vs predicted owl type
actual_predicted <- data.frame(actual = owls_test1$type,
                               predicted = predicted)

## Writing the actual and predicted values to a file for a particular iteration
##write.csv(file = "ActualVsPredicted",actual_predicted)

## Creating a compare variable which stores 1 if actual matches predicted else 0
actual_predicted$Compare <- ifelse(actual_predicted$actual == actual_predicted$predicted,1,0)

## Storing individual accuracies of the iterations for random split data
accuracy[a] <- (sum(actual_predicted$Compare == 1)/nrow(actual_predicted)) * 100

## printing the individual accuracies
message("Iteration",a)
print(accuracy[a])
}

## Plotting the individual accuracies of the 10 iterations
accuracy <- as.data.frame(accuracy)
ggplot(data = accuracy) + geom_point(mapping = aes(x = c(1:10), y = accuracy)) +
  geom_line(mapping = aes(x = c(1:10), y = accuracy)) + xlab("Iterations") +
  ggtitle("Accuracy per iterations")

## Calculating the average accuracy
average_accuracy <- (sum(accuracy)/1000) * 100
average_accuracy


