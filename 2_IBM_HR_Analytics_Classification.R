#Set Working Directory
setwd("C:/Users/utente/Documents/R/GitHub_Projects/Predict_Attrition_on_IBM_HR_Analytics_Dataset")

library(tictoc)
tic("Since Star_Time")

dir.create("figs") #save plots
dir.create("figs/Confusion Matrices") #save Confusion Matrices' plots
dir.create("figs/ROC Curves") # Save ROC Curves plots
dir.create("figs/Models' Performances") # Save Models' Performances plots
dir.create("figs/Variables' Importance") # Save Variables' Importance plots

#Import packages
library(caret)
library(tidyverse)
library(readxl)
library(writexl)
library(dplyr)
library(ggplot2)
library(forcats)
library(usefun)
library(MLeval)
library(plotROC)

#Import Dataset
library(readxl)
data <- read_excel("IBM HR Preprocessed.xlsx")
data <- as.data.frame(data)
str(data)

#Summary statistics
library(summarytools)
view(freq(data))
view(descr(data))
view(dfSummary(data))

#Remove ID variable
data <- data[,-9]

#Define proportion for training dataset
proportion <- 0.7

#Rename Target Variable
names(data)[2] <- "Y"

#Rename Target Variable's categories as positive or negative
data <- as.data.frame(data %>%
                      mutate(Y=case_when((Y == "Yes") ~ "Positive",
                                         (Y == "No") ~ "Negative")))

table(data$Y) #Dataset is unbalanced

#Subsampling
#data <- sample_n(data, round(0.25*nrow(data),digits = 0))

#Set number of variables you want to be displayed in Variables' Importance plots
N <- 15

#Set Tune Length for for tuning parameters
TuneLength <- 5
                          
#Coherce character columns to factors
char_col <- which(unlist(lapply(data, is.character)))
char_col
#data[,char_col] <- as.factor(data[,char_col])
data[,char_col] <- lapply(data[,char_col], factor)

#Coherce remaining columns to numeric
data[,-char_col] <- sapply(data[,-char_col], as.numeric)
data <- as.data.frame(data)

str(data)

#Centering and scaling
 numeric_col <- which(unlist(lapply(data, is.numeric)))
 numeric_col

 data <- data %>%
         mutate_if(is.numeric, scale)

 for (i in numeric_col) { #Verify if all numeric columns are standardized
   print(
     paste(
       names(data)[i],"has mean", round(mean(data[,i]),digits = 2),
       "and sd", sd(data[,i])

     )
   )
 }

###################### Plot Confusion Matrix Function ##########################
################################################################################

# cm = confusionMatrix(...)

draw_CM <- function(cm,Title) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), 
       type = "n", xlab="", ylab="",
       xaxt='n', yaxt='n')
  title(Title, cex.main = 2.0)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#72F069')
  text(195, 435, 'Negative', cex=1.8)
  rect(250, 430, 340, 370, col='#E84767')
  text(295, 435, 'Positive', cex=1.8)
  text(125, 370, 'Predicted', cex=1.9, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.9, font=2)
  rect(150, 305, 240, 365, col='#E84767')
  rect(250, 305, 340, 365, col='#72F069')
  text(140, 400, 'Negative', cex=1.8, srt=90)
  text(140, 335, 'Positive', cex=1.8, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.8, font=2, col='black')
  text(195, 335, res[2], cex=1.8, font=2, col='black')
  text(295, 400, res[3], cex=1.8, font=2, col='black')
  text(295, 335, res[4], cex=1.8, font=2, col='black')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", 
       main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.6, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 5), cex=1.6)
  text(30, 85, names(cm$byClass[2]), cex=1.6, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 5), cex=1.6)
  text(50, 85, names(cm$byClass[5]), cex=1.6, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 5), cex=1.6)
  text(70, 85, names(cm$byClass[6]), cex=1.6, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 5), cex=1.6)
  text(90, 85, names(cm$byClass[7]), cex=1.6, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 5), cex=1.6)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.6, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.6)
  text(70, 35, names(cm$overall[2]), cex=1.6, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.6)
} 


############################# ML Algorithms ####################################
################################################################################

#Data Splitting
set.seed(120496)  

inTrain <- createDataPartition(y = data$Y,
                               p = proportion,
                               list = FALSE)
training <- data[inTrain,]
dim(training)
table(training$Y)
testing  <- data[-inTrain,]
dim(testing)
table(testing$Y)

# Class proportion in training dataset for target variable
table(training$Y)
prop.table(table(training$Y))
################

#setup trainControl

#library(themis) #for SMOTE sampling for unbalanced datasets
#library(ROSE)   #for ROSE sampling for unbalanced datasets


fitControl <- trainControl(method = 'cv', 
                           number = 5,
                           verboseIter = FALSE,
                           classProbs = TRUE,
                           sampling = "down",
                           savePredictions = TRUE,
                           search = "grid")

                           


                        ###### Naive Bayes 
library(naivebayes)

grid_NB <-  expand.grid(laplace = seq(0,0.25,0.05),
                      usekernel = c(TRUE,FALSE),
                      adjust = seq(0.90,1.15,0.05))

tic("Naive Bayes")

#setup trainControl
NB <- train(Y ~ ., 
            data = training, 
            method = "naive_bayes",
            trControl= fitControl,
            #tuneGrid = grid_NB,
            tuneLength = TuneLength)



ELT <- toc()

sink("./Console output.txt",append = TRUE)
print(paste(ELT$msg,(ELT$toc - ELT$tic), "sec elapsed"))
print(paste(ELT$msg,(ELT$toc - ELT$tic)/60, "min elapsed"))
print(paste(ELT$msg,(ELT$toc - ELT$tic)/3600, "hours elapsed"))
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
sink()

NB
summary(NB)
NB$finalModel
NB$results
NB$bestTune

#ROC and AUC
x <- evalm(NB)
x$roc
ggsave(paste("ROC curve of ",NB$modelInfo$label,".png"),
       path = "./figs/ROC Curves")
x$stdres
y<- as.data.frame(x$stdres)
auc_NB <- y[13,1]
auc_NB

#Variables' Importance
VarImp_NB <- varImp(NB, scale = TRUE)
VarImp_NB
CM <- plot(VarImp_NB, 
           main = paste("Variables' Importance in", NB$modelInfo$label),
           type = "b", cex=1.3,top = N)
CM

png(filename = paste("./figs/Variables' Importance/",NB$modelInfo$label,".png"),
    width = 490, height = 580, units = "px", pointsize = 15)
CM
dev.off()

#Prediction
prediction_NB  <- predict(NB,testing)
prediction_NB

#Confusion matrix
confusion_matrix_NB <- confusionMatrix(prediction_NB,
                                       testing$Y,
                                       positive = "Positive")
confusion_matrix_NB     


draw_CM(confusion_matrix_NB,paste("Confusion Matrix in", NB$modelInfo$label))


png(filename = paste("./figs/Confusion Matrices/", NB$modelInfo$label,".png"),
    width = 900, height = 900, units = "px", pointsize = 25)
draw_CM(confusion_matrix_NB,paste("Confusion Matrix in", NB$modelInfo$label))
dev.off()

               #Support Vector Machines with Linear Kernel (SVMLK)

grid_SVMLK <-  expand.grid(C = c(seq(0.85,1.2,0.05),2,3,8,11))

library(kernlab)

tic("svmLinear")
SVMLK <- train(Y ~ .,
               data = training,
               method = "svmLinear",
               trControl= fitControl,
               #tuneGrid = grid_SVMLK,
               tuneLength = TuneLength)


ELT <- toc()

sink("./Console output.txt",append = TRUE)
print(paste(ELT$msg,(ELT$toc - ELT$tic), "sec elapsed"))
print(paste(ELT$msg,(ELT$toc - ELT$tic)/60, "min elapsed"))
print(paste(ELT$msg,(ELT$toc - ELT$tic)/3600, "hours elapsed"))
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
sink()

SVMLK
summary(SVMLK)
SVMLK$finalModel
SVMLK$results
SVMLK$bestTune


#ROC and AUC
x <- evalm(SVMLK)
x$roc
ggsave(paste("ROC curve of ",SVMLK$modelInfo$label,".png"),
       path = "./figs/ROC Curves")
x$stdres
y<- as.data.frame(x$stdres)
auc_SVMLK <- y[13,1]
auc_SVMLK

#Variables Importance
VarImp_SVMLK <- varImp(SVMLK, scale = TRUE)
VarImp_SVMLK
CM <- plot(VarImp_SVMLK, 
           main = paste("Variables' Importance in", SVMLK$modelInfo$label),
           type = "b", cex=1.3,top = N)
CM

png(filename = paste("./figs/Variables' Importance/",SVMLK$modelInfo$label,".png"),
    width = 490, height = 580, units = "px", pointsize = 15)
CM
dev.off()

#prediction
prediction_SVMLK <- predict(SVMLK,testing)
prediction_SVMLK

#Confusion matrix
confusion_matrix_SVMLK <- confusionMatrix(prediction_SVMLK,
                                          testing$Y,
                                          positive = "Positive")
confusion_matrix_SVMLK

draw_CM(confusion_matrix_SVMLK,paste("Confusion Matrix in", SVMLK$modelInfo$label))


png(filename = paste("./figs/Confusion Matrices/", SVMLK$modelInfo$label,".png"),
    width = 900, height = 900, units = "px", pointsize = 25)
draw_CM(confusion_matrix_SVMLK,paste("Confusion Matrix in", SVMLK$modelInfo$label))
dev.off()


        # Support Vector Machines with Radial Basis Function Kernel (SVMRK)
grid_SVMRK <-  expand.grid(C = c(seq(1.75,2.25,0.05),2,5,8,11))

#setup trainControl
tic("svmRadialCost")
SVMRK <- train(Y ~ ., 
               data = training, 
               method = "svmRadialCost",
               trControl= fitControl,
               #tuneGrid = grid_SVMRK,
               tuneLength = TuneLength)

ELT <- toc()

sink("./Console output.txt",append = TRUE)
print(paste(ELT$msg,(ELT$toc - ELT$tic), "sec elapsed"))
print(paste(ELT$msg,(ELT$toc - ELT$tic)/60, "min elapsed"))
print(paste(ELT$msg,(ELT$toc - ELT$tic)/3600, "hours elapsed"))
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
sink()

SVMRK
summary(SVMRK)
SVMRK$finalModel
SVMRK$results
SVMRK$bestTune

#ROC and AUC
x <- evalm(SVMRK)
x$roc
ggsave(paste("ROC curve of ",SVMRK$modelInfo$label,".png"),
       path = "./figs/ROC Curves")
x$stdres
y<- as.data.frame(x$stdres)
auc_SVMRK  <- y[13,1]
auc_SVMRK

#Variables Importance
VarImp_SVMRK <- varImp(SVMRK, scale = TRUE)
VarImp_SVMRK
CM <- plot(VarImp_NB, 
           main = paste("Variables' Importance in", SVMRK$modelInfo$label),
           type = "b", cex=1.3,top = N)
CM

png(filename = paste("./figs/Variables' Importance/",SVMRK$modelInfo$label,".png"),
    width = 490, height = 580, units = "px", pointsize = 15)
CM
dev.off()

#prediction
prediction_SVMRK <- predict(SVMRK,testing)
prediction_SVMRK


#Confusion matrix
confusion_matrix_SVMRK <- confusionMatrix(prediction_SVMRK,
                                          testing$Y,
                                          positive = "Positive")
confusion_matrix_SVMRK 

draw_CM(confusion_matrix_SVMRK,paste("Confusion Matrix in",SVMRK$modelInfo$label))


png(filename = paste("./figs/Confusion Matrices/", SVMRK$modelInfo$label,".png"),
    width = 900, height = 900, units = "px", pointsize = 25)
draw_CM(confusion_matrix_SVMRK,paste("Confusion Matrix in", SVMRK$modelInfo$label))
dev.off()


          # Support Vector Machines with Polynomial Kernel (SVMPL) 

 grid_SVMPL <-  expand.grid(C = c(seq(0.45,0.55,0.05),1,3),
                      degree = c(1,2,5,7),
                      scale = c(0,0.01,0.02,0.1,1,1.1))

#setup trainControl
tic("svmPoly")
SVMPL <- train(Y ~ ., 
               data = training, 
               method = "svmPoly",
               trControl= fitControl,
               #tuneGrid = grid_SVMPL,
               tuneLength = TuneLength)

ELT <- toc()

sink("./Console output.txt",append = TRUE)
print(paste(ELT$msg,(ELT$toc - ELT$tic), "sec elapsed"))
print(paste(ELT$msg,(ELT$toc - ELT$tic)/60, "min elapsed"))
print(paste(ELT$msg,(ELT$toc - ELT$tic)/3600, "hours elapsed"))
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
sink()

SVMPL
summary(SVMPL)
SVMPL$finalModel
SVMPL$results
SVMPL$bestTune

#ROC and AUC
x <- evalm(SVMPL)
x$roc
ggsave(paste("ROC curve of ",SVMPL$modelInfo$label,".png"),
       path = "./figs/ROC Curves")
x$stdres
y<- as.data.frame(x$stdres)
auc_SVMPL <- y[13,1]
auc_SVMPL


#Variables' Importance
VarImp_SVMPL <- varImp(SVMPL, scale=TRUE)
VarImp_SVMPL
CM <- plot(VarImp_NB, 
           main = paste("Variables' Importance in", SVMPL$modelInfo$label),
           type = "b", cex=1.3,top = N)
CM

png(filename = paste("./figs/Variables' Importance/",SVMPL$modelInfo$label,".png"),
    width = 490, height = 580, units = "px", pointsize = 15)
CM
dev.off()

#prediction
prediction_SVMPL <- predict(SVMPL,testing)
prediction_SVMPL

#Confusion matrix
confusion_matrix_SVMPL <- confusionMatrix(prediction_SVMPL,
                                          testing$Y, 
                                          positive = "Positive")
confusion_matrix_SVMPL

draw_CM(confusion_matrix_SVMPL,paste("Confusion Matrix in", SVMPL$modelInfo$label))


png(filename = paste("./figs/Confusion Matrices/", SVMPL$modelInfo$label,".png"),
    width = 900, height = 900, units = "px", pointsize = 25)
draw_CM(confusion_matrix_SVMPL,paste("Confusion Matrix in", SVMPL$modelInfo$label))
dev.off()

                             # Neural Network (NN)

library(nnet)
 grid_NN <-  expand.grid(size = c(seq(1,5,1),7,9),
                      decay = c(0,0.001,0.002,0.003,0.01,0.02,0.05,0.09,0.1,0.18,0.3,0.4))

#setup trainControl
tic("nnet")
NN <- train(Y ~ ., 
            data = training, 
            method = "nnet",
            trControl= fitControl,
            #tuneGrid = grid_NN,
            tuneLength = TuneLength)
ELT <- toc()

sink("./Console output.txt",append = TRUE)
print(paste(ELT$msg,(ELT$toc - ELT$tic), "sec elapsed"))
print(paste(ELT$msg,(ELT$toc - ELT$tic)/60, "min elapsed"))
print(paste(ELT$msg,(ELT$toc - ELT$tic)/3600, "hours elapsed"))
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
sink()

NN
summary(NN)
NN$finalModel
NN$results
NN$bestTune

#ROC and AUC
x <- evalm(NN)
x$roc
ggsave(paste("ROC curve of ",NN$modelInfo$label,".png"),
       path = "./figs/ROC Curves")
x$stdres
y<- as.data.frame(x$stdres)
auc_NN <- y[13,1]
auc_NN

#Variables' Importance
VarImp_NN <- varImp(NN, scale=TRUE)
VarImp_NN
CM <- plot(VarImp_NB, 
           main = paste("Variables' Importance in", NN$modelInfo$label),
           type = "b", cex=1.3,top = N)
CM

png(filename = paste("./figs/Variables' Importance/",NN$modelInfo$label,".png"),
    width = 490, height = 580, units = "px", pointsize = 15)
CM
dev.off()

#prediction
prediction_NN  <- predict(NN,testing)
prediction_NN 

#Confusion matrix
confusion_matrix_NN <- confusionMatrix(prediction_NN,
                                       testing$Y,
                                       positive = "Positive")
confusion_matrix_NN

draw_CM(confusion_matrix_NN,paste("Confusion Matrix in", NN$modelInfo$label))


png(filename = paste("./figs/Confusion Matrices/", NN$modelInfo$label,".png"),
    width = 900, height = 900, units = "px", pointsize = 25)
draw_CM(confusion_matrix_NN,paste("Confusion Matrix in", NN$modelInfo$label))
dev.off()


                     #Decision Tree (DT)
library(rpart)

grid_DT <-  expand.grid(cp = c(0.001,0.002,0.003,0.05,0.1,0.25,0.3,0.4,0.6))

#setup trainControl
tic("rpart")
DT <- train(Y ~ ., 
            data = training, 
            method = "rpart",
            trControl= fitControl,
            #tuneGrid = grid_DT,
            tuneLength = TuneLength)
ELT <- toc()

sink("./Console output.txt",append = TRUE)
print(paste(ELT$msg,(ELT$toc - ELT$tic), "sec elapsed"))
print(paste(ELT$msg,(ELT$toc - ELT$tic)/60, "min elapsed"))
print(paste(ELT$msg,(ELT$toc - ELT$tic)/3600, "hours elapsed"))
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
sink()

DT
summary(DT)
DT$finalModel
DT$results
DT$bestTune


#ROC and AUC
x <- evalm(DT)
x$roc
ggsave(paste("ROC curve of ",DT$modelInfo$label,".png"),
       path = "./figs/ROC Curves")
x$stdres
y<- as.data.frame(x$stdres)
auc_DT <- y[13,1]
auc_DT

#Variables' Importance
VarImp_DT <- varImp(DT, scale=TRUE)
VarImp_DT
CM <- plot(VarImp_NB, 
           main = paste("Variables' Importance in", DT$modelInfo$label),
           type = "b", cex=1.3,top = N)
CM

png(filename = paste("./figs/Variables' Importance/",DT$modelInfo$label,".png"),
    width = 490, height = 580, units = "px", pointsize = 15)
CM
dev.off()

#prediction
prediction_DT  <- predict(DT,testing)
prediction_DT

#Confusion matrix
confusion_matrix_DT <- confusionMatrix(prediction_DT,
                                       testing$Y,
                                       positive = "Positive")
confusion_matrix_DT

draw_CM(confusion_matrix_DT,paste("Confusion Matrix in", DT$modelInfo$label))


png(filename = paste("./figs/Confusion Matrices/", DT$modelInfo$label,".png"),
    width = 900, height = 900, units = "px", pointsize = 25)
draw_CM(confusion_matrix_DT,paste("Confusion Matrix in", DT$modelInfo$label))
dev.off()



                           ###### K- Nearest Neighbors (KNN)
grid_KNN <-  expand.grid(k = c(seq(1:22),25,28,34,50))

#setup trainControl
library(kknn)
tic("knn")
KNN <- train(Y ~ ., 
             data = training, 
             method = "knn",
             trControl= fitControl,
             #tuneGrid = grid_KNN,
             tuneLength = TuneLength)
ELT <- toc()

sink("./Console output.txt",append = TRUE)
print(paste(ELT$msg,(ELT$toc - ELT$tic), "sec elapsed"))
print(paste(ELT$msg,(ELT$toc - ELT$tic)/60, "min elapsed"))
print(paste(ELT$msg,(ELT$toc - ELT$tic)/3600, "hours elapsed"))
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
sink()

KNN
summary(KNN)
KNN$finalModel
KNN$results
KNN$bestTune


#ROC and AUC
x <- evalm(KNN)
x$roc
ggsave(paste("ROC curve of ",KNN$modelInfo$label,".png"),
       path = "./figs/ROC Curves")
x$stdres
y<- as.data.frame(x$stdres)
auc_KNN <- y[13,1]
auc_KNN

#Variables' Importance
VarImp_KNN<- varImp(KNN, scale=TRUE)
VarImp_KNN
CM <- plot(VarImp_NB, 
           main = paste("Variables' Importance in", KNN$modelInfo$label),
           type = "b", cex=1.3,top = N)
CM

png(filename = paste("./figs/Variables' Importance/",KNN$modelInfo$label,".png"),
    width = 490, height = 580, units = "px", pointsize = 15)
CM
dev.off()

#prediction
prediction_KNN  <- predict(KNN,testing)
prediction_KNN

#Confusion matrix
confusion_matrix_KNN <- confusionMatrix(prediction_KNN,
                                        testing$Y, 
                                        positive = "Positive")
confusion_matrix_KNN

draw_CM(confusion_matrix_KNN,paste("Confusion Matrix in", KNN$modelInfo$label))


png(filename = paste("./figs/Confusion Matrices/", KNN$modelInfo$label,".png"),
    width = 900, height = 900, units = "px", pointsize = 25)
draw_CM(confusion_matrix_KNN,paste("Confusion Matrix in", KNN$modelInfo$label))
dev.off()

                  ###### Boosted Logistic Regression

library(caTools)

grid_LB <-  expand.grid(nIter = c(25:45,71,81,100) )

#setup trainControl
tic("LB")
LB <- train(Y ~ ., 
             data = training, 
             method = "LogitBoost",
             trControl= fitControl,
             #tuneGrid = grid_LB,
             tuneLength = TuneLength)

ELT <- toc()

sink("./Console output.txt",append = TRUE)
print(paste(ELT$msg,(ELT$toc - ELT$tic), "sec elapsed"))
print(paste(ELT$msg,(ELT$toc - ELT$tic)/60, "min elapsed"))
print(paste(ELT$msg,(ELT$toc - ELT$tic)/3600, "hours elapsed"))
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
sink()

LB
summary(LB)
LB$finalModel
LB$results
LB$bestTune

#ROC and AUC
x <- evalm(LB)
x$roc
ggsave(paste("ROC curve of ",LB$modelInfo$label,".png"),
       path = "./figs/ROC Curves")
x$stdres
y<- as.data.frame(x$stdres)
auc_LB <- y[13,1]
auc_LB

#Variables' Importance
VarImp_LB <- varImp(LB, scale=TRUE)
VarImp_LB
CM <- plot(VarImp_NB, 
           main = paste("Variables' Importance in", LB$modelInfo$label),
           type = "b", cex=1.3,top = N)
CM

png(filename = paste("./figs/Variables' Importance/",LB$modelInfo$label,".png"),
    width = 490, height = 580, units = "px", pointsize = 15)
CM
dev.off()

#prediction
prediction_LB  <- predict(LB,testing)
prediction_LB

#Confusion matrix
confusion_matrix_LB <- confusionMatrix(prediction_LB,
                                        testing$Y, 
                                        positive = "Positive")
confusion_matrix_LB


draw_CM(confusion_matrix_LB,paste("Confusion Matrix in", LB$modelInfo$label))


png(filename = paste("./figs/Confusion Matrices/", LB$modelInfo$label,".png"),
    width = 900, height = 900, units = "px", pointsize = 25)
draw_CM(confusion_matrix_LB,paste("Confusion Matrix in", LB$modelInfo$label))
dev.off()

                         ###### Random Forest

library(randomForest)

grid_RF <-  expand.grid(mtry = c(10:20,32,38,41,51,60) )

#setup trainControl
tic("rf")
RF <- train(Y ~ ., 
            data = training, 
            method = "rf",
            trControl= fitControl,
            #tuneGrid = grid_RF,
            tuneLength = TuneLength)

ELT <- toc()

sink("./Console output.txt",append = TRUE)
print(paste(ELT$msg,(ELT$toc - ELT$tic), "sec elapsed"))
print(paste(ELT$msg,(ELT$toc - ELT$tic)/60, "min elapsed"))
print(paste(ELT$msg,(ELT$toc - ELT$tic)/3600, "hours elapsed"))
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
sink()

RF
summary(RF)
RF$finalModel
RF$results
RF$bestTune

#ROC and AUC
x <- evalm(RF)
x$roc
ggsave(paste("ROC curve of ",RF$modelInfo$label,".png"),
       path = "./figs/ROC Curves")
x$stdres
y<- as.data.frame(x$stdres)
auc_RF <- y[13,1]
auc_RF

#Variables' Importance
VarImp_RF <- varImp(RF, scale=TRUE)
VarImp_RF
CM <- plot(VarImp_NB, 
           main = paste("Variables' Importance in", RF$modelInfo$label),
           type = "b", cex=1.3,top = N)
CM

png(filename = paste("./figs/Variables' Importance/",RF$modelInfo$label,".png"),
    width = 490, height = 580, units = "px", pointsize = 15)
CM
dev.off()

#prediction
prediction_RF  <- predict(RF,testing)
prediction_RF

#Confusion matrix
confusion_matrix_RF <- confusionMatrix(prediction_RF,
                                       testing$Y, 
                                       positive = "Positive")


confusion_matrix_RF


draw_CM(confusion_matrix_RF,paste("Confusion Matrix in", RF$modelInfo$label))


png(filename = paste("./figs/Confusion Matrices/", RF$modelInfo$label,".png"),
    width = 900, height = 900, units = "px", pointsize = 25)
draw_CM(confusion_matrix_RF,paste("Confusion Matrix in", RF$modelInfo$label))
dev.off()



############################# Models Resampling ####################################
################################################################################

resamps <- resamples(list(NaiveBayes = NB,
                          svmLinear = SVMLK,
                          svmRadial = SVMRK,
                          svmPoly = SVMPL,
                          nnet = NN,
                          rpart = DT,
                          knn = KNN,
                          logistic = LB,
                          randomForest = RF))
resamps$timings

Timings <- data.frame(Models = rownames(resamps$timings),
                      Time = resamps$timings[,1])


resamps <- data.frame(Models = c("NaiveBayes",
                                 "svmLinear",
                                 "svmRadial",
                                 "svmPoly",
                                 "nnet",
                                 "rpart",
                                 "knn",
                                 "logistic",
                                 "randomForest"),
                      Accuracy = c(confusion_matrix_NB$overall[1],
                                   confusion_matrix_SVMLK$overall[1],
                                   confusion_matrix_SVMRK$overall[1],
                                   confusion_matrix_SVMPL$overall[1],
                                   confusion_matrix_NN$overall[1],
                                   confusion_matrix_DT$overall[1],
                                   confusion_matrix_KNN$overall[1],
                                   confusion_matrix_LB$overall[1],
                                   confusion_matrix_RF$overall[1]),
                      Kappa = c(confusion_matrix_NB$overall[2],
                                confusion_matrix_SVMLK$overall[2],
                                confusion_matrix_SVMRK$overall[2],
                                confusion_matrix_SVMPL$overall[2],
                                confusion_matrix_NN$overall[2],
                                confusion_matrix_DT$overall[2],
                                confusion_matrix_KNN$overall[2],
                                confusion_matrix_LB$overall[2],
                                confusion_matrix_RF$overall[2]),
                      Sensitivity = c(confusion_matrix_NB$byClass[1],
                                      confusion_matrix_SVMLK$byClass[1],
                                      confusion_matrix_SVMRK$byClass[1],
                                      confusion_matrix_SVMPL$byClass[1],
                                      confusion_matrix_NN$byClass[1],
                                      confusion_matrix_DT$byClass[1],
                                      confusion_matrix_KNN$byClass[1],
                                      confusion_matrix_LB$byClass[1],
                                      confusion_matrix_RF$byClass[1]),
                      Specificity = c(confusion_matrix_NB$byClass[2],
                                      confusion_matrix_SVMLK$byClass[2],
                                      confusion_matrix_SVMRK$byClass[2],
                                      confusion_matrix_SVMPL$byClass[2],
                                      confusion_matrix_NN$byClass[2],
                                      confusion_matrix_DT$byClass[2],
                                      confusion_matrix_KNN$byClass[2],
                                      confusion_matrix_LB$byClass[2],
                                      confusion_matrix_RF$byClass[2]),
                      Precision = c(confusion_matrix_NB$byClass[5],
                                    confusion_matrix_SVMLK$byClass[5],
                                    confusion_matrix_SVMRK$byClass[5],
                                    confusion_matrix_SVMPL$byClass[5],
                                    confusion_matrix_NN$byClass[5],
                                    confusion_matrix_DT$byClass[5],
                                    confusion_matrix_KNN$byClass[5],
                                    confusion_matrix_LB$byClass[5],
                                    confusion_matrix_RF$byClass[5]),
                      F1 = c(confusion_matrix_NB$byClass[7],
                             confusion_matrix_SVMLK$byClass[7],
                             confusion_matrix_SVMRK$byClass[7],
                             confusion_matrix_SVMPL$byClass[7],
                             confusion_matrix_NN$byClass[7],
                             confusion_matrix_DT$byClass[7],
                             confusion_matrix_KNN$byClass[7],
                             confusion_matrix_LB$byClass[7],
                             confusion_matrix_RF$byClass[7]),
                      AUC = c(auc_NB,
                              auc_SVMLK,
                              auc_SVMRK,
                              auc_SVMPL,
                              auc_NN,
                              auc_DT,
                              auc_NN,
                              auc_LB,
                              auc_RF))

#Export Results in a .xlsx file
library(writexl)
write_xlsx(resamps,".\\Models' results.xlsx")


############################### Model's Perfomances Plots ######################
################################################################################

#Plot accuracy
ggplot(data=resamps,
       aes(x=Models, y=Accuracy, fill=Models)) +
  geom_bar(stat="identity", position=position_dodge(),
           color = "black")+
  geom_text(aes(label=paste(round(Accuracy*100,digits =3),"%")),
            vjust = 0.5,
            hjust = 1.8,
            color="black",
            position = position_dodge(0.9), 
            size=3.5)+
  scale_fill_brewer(palette="Paired")+
  theme_bw()+
  coord_flip()+
  theme(legend.position = "none")+
  ggtitle("Models' Accuracy")+
  ylab("")

ggsave("Accuracy.png",
       path = "./figs/Models' Performances")

#Plot Kappa
ggplot(data=resamps,
       aes(x=Models, y=Kappa, fill=Models)) +
  geom_bar(stat="identity", position=position_dodge(),
           color = "black")+
  geom_text(aes(label=paste(round(Kappa,digits =3))),
            vjust = 0.5,
            hjust = 1.8,
            color="black",
            position = position_dodge(0.9), 
            size=3.5)+
  scale_fill_brewer(palette="Paired")+
  theme_bw()+
  coord_flip()+
  theme(legend.position = "none")+
  ggtitle("Models' Kappa")+
  ylab("")

ggsave("Kappa.png",
       path = "./figs/Models' Performances")

#Plot Sensitivity
ggplot(data=resamps,
       aes(x=Models, y=Sensitivity, fill=Models)) +
  geom_bar(stat="identity", position=position_dodge(),
           color = "black")+
  geom_text(aes(label=paste(round(Sensitivity,digits =3))),
            vjust = 0.5,
            hjust = 1.8,
            color="black",
            position = position_dodge(0.9), 
            size=3.5)+
  scale_fill_brewer(palette="Paired")+
  theme_bw()+
  coord_flip()+
  theme(legend.position = "none")+
  ggtitle("Models' Sensitivity")+
  ylab("")

ggsave("Sensitivity.png",
       path = "./figs/Models' Performances")


#Plot Specificity
ggplot(data=resamps,
       aes(x=Models, y=Specificity, fill=Models)) +
  geom_bar(stat="identity", position=position_dodge(),
           color = "black")+
  geom_text(aes(label=paste(round(Specificity,digits =3))),
            vjust = 0.5,
            hjust = 1.8,
            color="black",
            position = position_dodge(0.9), 
            size=3.5)+
  scale_fill_brewer(palette="Paired")+
  theme_bw()+
  coord_flip()+
  theme(legend.position = "none")+
  ggtitle("Models' Specificity")+
  ylab("")

ggsave("Specificity.png",
       path = "./figs/Models' Performances")

#Plot Precision
ggplot(data=resamps,
       aes(x=Models, y=Precision, fill=Models)) +
  geom_bar(stat="identity", position=position_dodge(),
           color = "black")+
  geom_text(aes(label=paste(round(Precision,digits =3))),
            vjust = 0.5,
            hjust = 1.8,
            color="black",
            position = position_dodge(0.9), 
            size=3.5)+
  scale_fill_brewer(palette="Paired")+
  theme_bw()+
  coord_flip()+
  theme(legend.position = "none")+
  ggtitle("Models' Precision")+
  ylab("")

ggsave("Precision.png",
       path = "./figs/Models' Performances")


#Plot F1
ggplot(data=resamps,
       aes(x=Models, y=F1, fill=Models)) +
  geom_bar(stat="identity", position=position_dodge(),
           color = "black")+
  geom_text(aes(label=paste(round(F1,digits =3))),
            vjust = 0.5,
            hjust = 1.8,
            color="black",
            position = position_dodge(0.9), 
            size=3.5)+
  scale_fill_brewer(palette="Paired")+
  theme_bw()+
  coord_flip()+
  theme(legend.position = "none")+
  ggtitle("Models' F1")+
  ylab("")

ggsave("F1.png",
       path = "./figs/Models' Performances")

#Plot AUC
ggplot(data=resamps,
       aes(x=Models, y=AUC, fill=Models)) +
  geom_bar(stat="identity", position=position_dodge(),
           color = "black")+
  geom_text(aes(label=paste(round(AUC,digits =3))),
            vjust = 0.5,
            hjust = 1.8,
            color="black",
            position = position_dodge(0.9), 
            size=3.5)+
  scale_fill_brewer(palette="Paired")+
  theme_bw()+
  coord_flip()+
  theme(legend.position = "none")+
  ggtitle("Models' AUC")+
  ylab("")

ggsave("AUC.png",
       path = "./figs/Models' Performances")

############################### Models' Time Elapsed Plot #####################
################################################################################
Custom_Palette <- c("#E64B35B2","#4DBBD5B2","#00A087B2","#3C5488B2",
                    "#F39B7FB2","#8491B4B2","#91D1C2B2","#DC0000B2" 
                    ,"#7E6148B2")


Timings$prop <- Timings$Time/sum(Timings$Time) #percentages


#Plot Time elapsed (%)
ggplot(data=Timings,
       aes(x=reorder(Models,-prop),y=prop, fill=Models)) +
  geom_bar(stat="identity", 
           position=position_dodge(),
           color = "black")+
  geom_text(aes(label=paste(round(prop,digits =4)*100,"%")),
            vjust = -0.3,
            hjust = 0.5,
            color="black",
            position = position_dodge(0.8), 
            size=3.5)+
  scale_fill_manual(values = Custom_Palette)+
  theme_bw()+
  #coord_flip()+
  theme(legend.position = "none")+
  ggtitle("Models' Time Elapsed")+
  ylab("")+
  xlab("")+
  annotate(geom="text",x=length(Timings$Models)-0.5,
           y=max(Timings$prop)-min(Timings$prop),
           label=paste("Total algorithms'\n elapsed time \n",round(sum(Timings$Time),2)),
           color="black", size = 5)


ggsave("Time Elapsed.png",
       path = "./")

######################### Script Exectution time ###############################
################################################################################
ELT <- toc()

######################### Export Model's Performances ##########################
################################################################################

sink("./Console output.txt",append = TRUE)
print(paste(ELT$msg,(ELT$toc - ELT$tic), "sec elapsed"))
print(paste(ELT$msg,(ELT$toc - ELT$tic)/60, "min elapsed"))
print(paste(ELT$msg,(ELT$toc - ELT$tic)/3600, "hours elapsed"))
sink()


#Export Models' performances on an output .txt file

sink("./Console output.txt",append = TRUE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)

print(paste("############## Naive Bayes","#################",
            "##############################################"))
confusion_matrix_NB$table
confusion_matrix_NB$overall
confusion_matrix_NB$byClass
NB$bestTune
NB$results
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)


print(paste("############## SVM Linear Kernel","#################",
            "####################################################"))
confusion_matrix_SVMLK$table
confusion_matrix_SVMLK$overall
confusion_matrix_SVMLK$byClass
SVMLK$bestTune
SVMLK$results
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)


print(paste("############## SVM Radial Kernel","#################",
            "####################################################"))
confusion_matrix_SVMRK$table
confusion_matrix_SVMRK$overall
confusion_matrix_SVMRK$byClass
SVMRK$bestTune
SVMRK$results
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)

print(paste("############## SVM Polynomial Kernel","#################",
            "########################################################"))
confusion_matrix_SVMPL$table
confusion_matrix_SVMPL$overall
confusion_matrix_SVMPL$byClass
SVMPL$bestTune
SVMPL$results
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)


print(paste("############## Neural Network","#################",
            "#################################################"))
confusion_matrix_NN$table
confusion_matrix_NN$overall
confusion_matrix_NN$byClass
NN$bestTune
NN$results
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)


print(paste("############## Decision Tree","#################",
            "################################################"))
confusion_matrix_DT$table
confusion_matrix_DT$overall
confusion_matrix_DT$byClass
DT$bestTune
DT$results
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)


print(paste("############## K- Nearest Neighbors","#################",
            "#######################################################"))
confusion_matrix_KNN$table
confusion_matrix_KNN$overall
confusion_matrix_KNN$byClass
KNN$bestTune
KNN$results
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)


print(paste("############## Boosted Logistic Regression","#################",
            "##############################################################"))
confusion_matrix_LB$table
confusion_matrix_LB$overall
confusion_matrix_LB$byClass
LB$bestTune
LB$results
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)

print(paste("############## Random Forest","#################",
            "##############################################################"))
confusion_matrix_RF$table
confusion_matrix_RF$overall
confusion_matrix_RF$byClass
RF$bestTune
RF$results
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)
print_empty_line(html.output = FALSE)

sink()


