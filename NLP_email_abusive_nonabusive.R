
# Load Libraries
library(tm)
library(plyr)
library(class)
library(caret)
library(e1071)
library(knitr)

# Read data
rawdata <- email1_full_clean2[,(1:2)]
names(rawdata) <- c("Class","content")
kable(rawdata[1:8,])
str(rawdata)
rawdata$content<-as.character(rawdata$content)
#Preprocessing
# Find total number of characters in each SMS
NumberOfChar <- as.numeric(lapply(rawdata$content,FUN=nchar))
NumberOfChar
# Find number of numeric digits in each SMS

number.digits <- function(vect) {
  length(as.numeric(unlist(strsplit(gsub("[^0-9]", "", unlist(vect)), ""))))
}

NumberOfDigits <- as.numeric(lapply(rawdata$content,FUN=number.digits))
NumberOfDigits
# Function to clean text in the SMS

clean.text = function(x)
{ 
  # tolower
  x = tolower(x)
  # remove punctuation
  x = gsub("[[:punct:]]", "", x)
  # remove numbers
  x = gsub("[[:digit:]]", "", x)
  # remove tabs
  x = gsub("[ |\t]{2,}", "", x)
  # remove blank spaces at the beginning
  x = gsub("^ ", "", x)
  # remove blank spaces at the end
  x = gsub(" $", "", x)
  # remove common words
  x = removeWords(x,stopwords("en"))
  return(x)
}

cleanText <- clean.text(rawdata$content)

# Build Corpus
corpus <- Corpus(VectorSource(cleanText))

# Build Term Document Matrix
tdm <- DocumentTermMatrix(corpus)

# Convert TDM to Dataframe
tdm.df <- as.data.frame(data.matrix(tdm),stringsAsFactors=FALSE)

# Remove features with total frequency less than 3
tdm.new <- tdm.df[,colSums(tdm.df) > 2]
#Split Data:
# Prepare final data with TDM, NumberofChar, NumberOfDigits as features
  
cleandata <- cbind("Class" = rawdata$Class, NumberOfChar, NumberOfDigits, tdm.new)
str(cleandata)
# Split Data into training (80%) and testing(20%) datasets

set.seed(1234)
inTrain <- createDataPartition(cleandata$Class,p=0.8,list=FALSE)
train <- cleandata[inTrain,]
test <- cleandata[-inTrain,]

#Build SVM Models
## Linear Kernel
svm.linear <- svm(Class~., data=train, scale=FALSE, kernel='linear')
pred.linear <- predict(svm.linear, test[,-1])
linear <- confusionMatrix(pred.linear,test$Class)

## Linear Kernel
svm.poly <- svm(Class~., data=train, scale=FALSE, kernel='polynomial')
pred.poly <- predict(svm.poly, test[,-1])
poly <- confusionMatrix(pred.poly,test$Class)

## Radial Basis Kernel
svm.radial <- svm(Class~., data=train, scale=FALSE, kernel='radial')
pred.radial <- predict(svm.radial,test[,-1])
radial <- confusionMatrix(pred.radial,test$Class)

## Sigmoid Kernel
svm.sigmoid <- svm(Class~., data=train, scale=FALSE, kernel='sigmoid')
pred.sigmoid <- predict(svm.sigmoid,test[,-1])
sigmoid <- confusionMatrix(pred.sigmoid,test$Class)

#Accuracies
Kernels <- c("Linear","Polynomial","Radial Basis","Sigmoid")
Accuracies <- round(c(linear$overall[1],poly$overall[1],radial$overall[1],sigmoid$overall[1]),4)
acc <- cbind(Kernels,Accuracies)
kable(acc,row.names=FALSE)
conf.mat <- confusionMatrix(pred.sigmoid, test$Class)
conf.mat

#Accuracies-1
Kernel <- c("Linear")
Accuracy <- (linear$overall[1])
acc <- cbind(Kernel,Accuracy)
kable(acc,row.names=FALSE)
conf.mat <- confusionMatrix(pred.linear, test$Class)
conf.mat

save(linear,file = "SVM_linear_model.rda")
save(tdm.new, file = "linear_dtm.rda")
save(number.digits, file = "numberofdigits.rda")
save(clean.text, file = "textclean.rda")
