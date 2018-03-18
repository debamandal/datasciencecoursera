

##Prediction Assignment

This is the final course project for Practical Machine Learning at Coursera.
The project is about HAR - Human Activity Recognition , using different wearable accelerometers like  Jawbone Up, Nike FuelBand, and Fitbit. 
Few test subjects are made to perform some exercises in correct and incorrect manner, wearing different accelerometers.
Based on the train dataset generated, we need to predict the test cases.
The Aim of the project is to predict the “classe” variable for test subjects who performed some exercises.

##Data Sources
The training data for this project are available here:  https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
The test data are available here:  https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. 

##Code and Logic for Prediction
Loading the Data
setwd("C:/R/data/coursera/Machine_learning")
library(caret)
#Load the dataset
training <- read.csv("pml-training.csv",na.strings = "NA")
testing  <- read.csv("pml-testing.csv",na.strings = "NA")

##Cleaning the Data
After loading the data , it needs to be examined and cleaned. Looking at the structure of the data shows many columns containing mostly “NA” values.  Also columns with near zero variance also need to be removed.
At each  cleansing step I used the str function to look at the data.
#Cleaning Data and reducing predictors
#Look at the data structure
str(training)

#Remove the columns that are mostly NA
AllNA    <- sapply(training, function(x) mean(is.na(x))) > 0.95
training <- training[, AllNA==FALSE]
testing  <- testing[, AllNA==FALSE]
#Look at the data structure
str(training)
#Now the datasets have 93 variables 
#Remove columns with near zero variance
NonVar <- nearZeroVar(training,saveMetrics=TRUE)
training <- training[,!NonVar$nzv]
testing <- testing[,!NonVar$nzv]
#Look at the data structure
str(training)
#Now the datasets have 59 variables 
#Remove names & other variables not associated with prediction
colRm_train <- c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","num_window")
colRm_test <- c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","num_window","problem_id")
training <- training[,!(names(training) %in% colRm_train)]
testing <- testing[,!(names(testing) %in% colRm_test)]
#Now the datasets have 52 variables 
The training dataset is divided into 2 parts, training and validation, to verify the accuracy of the prediction model.
#Partition Training set into train and validation
inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
training_clean <- training[inTrain,]
validation_clean <- training[-inTrain,]
Multiple machine learning algorithms will be used , and the one with the best accuracy will be used on the test dataset.

##Algorithm 1: Random Forest

#Fit Random forest 
set.seed(1414)
rfFit <- train(classe ~ ., method = "rf", data = training_clean, importance = T, trControl = trainControl(method = "cv", number = 4))
validation_pred <- predict(rfFit, newdata=validation_clean)
# Check model performance
confusionMatrix(validation_pred,validation_clean$classe)
#Random Forest prediction statistics
Confusion Matrix and Statistics
          Reference
Prediction    A    B    C    D    E
         A 1674   12    0    0    0
         B    0 1127    6    0    0
         C    0    0 1019   18    0
         D    0    0    1  943    0
         E    0    0    0    3 1082

Overall Statistics
                                          
   Accuracy : 0.9932          
   95% CI : (0.9908, 0.9951)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9914          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   0.9895   0.9932   0.9782   1.0000
Specificity            0.9972   0.9987   0.9963   0.9998   0.9994
Pos Pred Value         0.9929   0.9947   0.9826   0.9989   0.9972
Neg Pred Value         1.0000   0.9975   0.9986   0.9957   1.0000
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2845   0.1915   0.1732   0.1602   0.1839
Detection Prevalence   0.2865   0.1925   0.1762   0.1604   0.1844
Balanced Accuracy      0.9986   0.9941   0.9947   0.9890   0.9997


##Algorithm 2: Boosted Model
#Fit Boosted Model
set.seed(1414)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitGBM  <- train(classe ~ ., data=training_clean, method = "gbm",
                    trControl = controlGBM, verbose = FALSE)
predictGBM <- predict(modFitGBM, newdata=validation_clean)
# Check model performance
confusionMatrix(predictGBM, validation_clean$classe)
#Boosted Model prediction Statistics
Confusion Matrix and Statistics
  Reference
Prediction    
               A    B    C    D    E
         A 1648   38    0    1    3
         B   15 1075   40    3   12
         C    8   22  975   30    7
         D    3    4    9  924   10
         E    0    0    2    6 1050

Overall Statistics
                                          
               Accuracy : 0.9638          
                 95% CI : (0.9587, 0.9684)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9542          
 Mcnemar's Test P-Value : 4.026e-08       

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9845   0.9438   0.9503   0.9585   0.9704
Specificity            0.9900   0.9853   0.9862   0.9947   0.9983
Pos Pred Value         0.9751   0.9389   0.9357   0.9726   0.9924
Neg Pred Value         0.9938   0.9865   0.9895   0.9919   0.9934
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2800   0.1827   0.1657   0.1570   0.1784
Detection Prevalence   0.2872   0.1946   0.1771   0.1614   0.1798
Balanced Accuracy      0.9872   0.9645   0.9683   0.9766   0.9844


##Algorithm 3: LDA Model

#Fit lda Model
set.seed(1414)
controlLDA <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitLDA  <- train(classe ~ ., data=training_clean, method = "lda",
                    trControl = controlLDA, verbose = FALSE)
predictLDA <- predict(modFitLDA, newdata=validation_clean)
# Check model performance
confusionMatrix(predictLDA, validation_clean$classe)
#LDA Model prediction Statistics
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1346  178   87   50   36
         B   51  733  108   37  177
         C  144  125  703  124  104
         D  126   46  101  720   92
         E    7   57   27   33  673

Overall Statistics
                                         
               Accuracy : 0.7094         
                 95% CI : (0.6976, 0.721)
    No Information Rate : 0.2845         
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.6326         
 Mcnemar's Test P-Value : < 2.2e-16      

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.8041   0.6435   0.6852   0.7469   0.6220
Specificity            0.9166   0.9214   0.8977   0.9258   0.9742
Pos Pred Value         0.7932   0.6627   0.5858   0.6636   0.8444
Neg Pred Value         0.9217   0.9150   0.9311   0.9492   0.9196
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2287   0.1246   0.1195   0.1223   0.1144
Detection Prevalence   0.2884   0.1879   0.2039   0.1844   0.1354
Balanced Accuracy      0.8604   0.7825   0.7915   0.8364   0.7981
Comparison of the 3 outputs
Random Forest Accuracy : 0.9932          
Boost Model Accuracy : 0.9638
LDA Model Accuracy : 0.7094    

##Random Forest is chosen to predict the test set results
#actual Prediction
predictTEST <- predict(rfFit, newdata=testing)
predictTEST     
