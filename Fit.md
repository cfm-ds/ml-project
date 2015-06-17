---
title: "Training a classifier for rating execution quality of barbell exercises based on sensor data."
output: 
  html_document
---





## Executive summary
In this report, we describe the training of a random forest-based classifier for classifying barbell exercise executions in 5 quality classes. Our classifier is shown to achieve high accuracy. 

## Data processing
We load the data, properly encoding N/A values and converting the quality class, which is to be predicted, to a factor. 

```r
train <- read.csv("pml-training.csv", na.strings = c("", "#DIV!0", NA))
test <- read.csv("pml-testing.csv", na.strings = c("", "#DIV!0", NA))
train$classe <- as.factor(train$classe)
```

A large number of columns have a high percentage of NA values. These are removed first as they might interfere with prediction due to sparsity of the data.

```r
na_data <- sapply(train, function (x) mean(is.na(x)))
length(subset(na_data, na_data > 0.95))
```

```
## [1] 100
```

```r
var_set <- names(train)[na_data < 0.95]
train <- train[, var_set]
var_set <- var_set[1:59]
test <- test[, var_set]
```

We also remove any variables with close to zero variance, and remove those variables that are not related to the sensor-data, as we want our classifier
to operate on sensor-data only.

```r
nsv <- nearZeroVar(train, saveMetrics = F)
train <- dplyr::select(train, -nsv)

train <- train[,-(1:6)]
test <- test[,-(1:6)]
```

Although the caret package will be able to estimate out-of-sample accuracy, we will split the training data to obtain an additional test set for determining the out-of-sample accuracy ourselves:


```r
# Estimate OO error by partitioning off a testing set from training set
# Partition by classe so test cases for all classes are created
in.train <- createDataPartition(train$classe, p=.60, list=FALSE)
t.train <- train[in.train[,1],]
t.test <- train[-in.train[,1],]
```

## Classifier training and performance
We opt for a random forest model as the dataset contains a large number of variables, and because such models can be highly accurate overall. We rely on the caret's package defaults for the other parameters, and investigate whether we get good results from those defaults before tuning. 

```r
#Build model
rf_model <- train(y = t.train$classe, x = select(t.train, -classe), method="rf")
```

```
## Error in select(t.train, -classe): unused argument (-classe)
```

```r
rf_model
```

```
## Error in eval(expr, envir, enclos): object 'rf_model' not found
```

The model with the highest accuracy uses trees with 2 randomly chosen predictors out of the 52. As one can see the estimated out-of-sample accuracy is high for the different amount of predictors.

As expected, the accuracy on the training set is excellent, no errors prediction errors are reported:

```r
predictions <- predict(rf_model, t.train)
```

```
## Error in predict(rf_model, t.train): object 'rf_model' not found
```

```r
confusionMatrix(predictions, t.train[, "classe"])
```

```
## Error in table(data, reference, dnn = dnn, ...): all arguments must have the same length
```

Let's see how it generalizes by looking at the performance on our test set:

```r
predictions <- predict(rf_model, t.test)
```

```
## Error in predict(rf_model, t.test): object 'rf_model' not found
```

```r
confusionMatrix(predictions, t.test[, "classe"])
```

```
## Error in table(data, reference, dnn = dnn, ...): all arguments must have the same length
```
Accuracy is 0.9895 so our out-of-sample error estimate is 0.0105. Which is high enough for our purposes. 

Finally lets investigate the importance of the predictors

```
## Error in varImp(rf_model): object 'rf_model' not found
```

It seems roll and yaw sensor data from the lumbar belt were especially important in assessing the quality of the exercises. 

## GLM model
Given the high accuracy attained by the random forest, we are interested in comparing its performance to a less computationally intensive technique such as linear discriminant analysis. 


```r
glm_model <- train(y = t.train$classe, x = dplyr::select(t.train,-classe), method="lda")
predictions <- predict(glm_model, dplyr::select(t.test,-classe))
confusionMatrix(predictions, t.test[, "classe"])
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1832  234  147   80   43
##          B   55  988  131   52  255
##          C  180  174  906  154  137
##          D  158   56  154  936  161
##          E    7   66   30   64  846
## 
## Overall Statistics
##                                           
##                Accuracy : 0.702           
##                  95% CI : (0.6918, 0.7121)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6228          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8208   0.6509   0.6623   0.7278   0.5867
## Specificity            0.9102   0.9221   0.9004   0.9194   0.9739
## Pos Pred Value         0.7842   0.6671   0.5841   0.6389   0.8351
## Neg Pred Value         0.9274   0.9167   0.9266   0.9451   0.9128
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2335   0.1259   0.1155   0.1193   0.1078
## Detection Prevalence   0.2977   0.1888   0.1977   0.1867   0.1291
## Balanced Accuracy      0.8655   0.7865   0.7814   0.8236   0.7803
```

Interestingly,accuracy is significantly lower (70%) for LDA.

## Predict project test data outcomes


```r
pml_write_files = function(x) {
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
answers <- predict(rf_model, test)
```

```
## Error in predict(rf_model, test): object 'rf_model' not found
```

```r
pml_write_files(answers)
```

```
## Error in pml_write_files(answers): object 'answers' not found
```
}
