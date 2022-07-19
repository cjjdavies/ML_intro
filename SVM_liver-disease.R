## ----setup, include=FALSE------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE, warning = FALSE, message = FALSE)


## ----load-packages-------------------------------------------------------------------------------------------------
library(knitr)
library(dplyr) # data wrangling
library(forcats)
library(ggplot2) # graphics
library(rsample) # data splitting

library(caret) # classification and regression
library(kernlab) # for fitting SVM

library(vip) # variable importance plots



## ----load-dataset--------------------------------------------------------------------------------------------------
# Load dataset, using dataset title, specify the first row as column names, 
# first column as row names, and the comma separator to delimit characters
hepC <- read.csv("hepatitisC_dataset.csv", header = T, row.names = 1, sep = ",")

# Check the structure of the dataset, showing the variable names, types and values
str(hepC)

# Inspect a summary statistics, checking for anomalies and missing values
summary(hepC)



## ----data-org------------------------------------------------------------------------------------------------------

# Rename factor levels to rename 0=Blood Donor and 0s=suspect Blood Donor as 0,
# and 1=Hepatitis, 2=Fibrosis, and 3=Cirrhosis as 1. This is important to be
# able to implement SVM.

hepC$Category <- recode(hepC$Category,
       "1=Hepatitis" = "Hepatitis",
       "2=Fibrosis" = "Hepatitis",
       "3=Cirrhosis" = "Hepatitis",
       "0=Blood Donor" = "Blood",
       "0s=suspect Blood Donor" = "Blood",
       .default = levels(hepC$Category)) # retains the original structure

# Specify 'Category' as a factor with two levels
hepC$Category <- factor(hepC$Category)

# Remove rows with missing data
hepC_narm <- hepC[complete.cases(hepC),]


## ----predictors----------------------------------------------------------------------------------------------------
# Filter out age and sex from the dataframe, using the blood sample data as
# predictors, and the Category as the dependent variable
hepC_sub <- hepC_narm %>% 
  select (-c(Age, Sex))



## ----data-splitting------------------------------------------------------------------------------------------------
# Set seed for reproducibility
set.seed(1234)

# Training data set at 70% training data and 30% testing data
hepC_sub_split <- initial_split(hepC_sub, prop = 0.7, strata = "Category")
hepC_train <- training(hepC_sub_split)
hepC_test <- testing(hepC_sub_split)



## ----model-spec----------------------------------------------------------------------------------------------------

set.seed(1234)  # for reproducibility

# trainControl function for cross-validation and estimation of class
# probabilities
ctrl <- trainControl(
  method = "cv", 
  number = 10, 
  classProbs = TRUE,
  savePredictions = TRUE,
  summaryFunction = twoClassSummary  # needed for AUC/ROC
)

# Category is specified as a two-level factor variable for classification purposes
# The radial method is considered the best starting point to train the model,
# and can be refined with more appropriate methods in later steps

# Build the SVM
set.seed(1234)  # for reproducibility

hepC_sub_auc <- train(
  Category ~ ., 
  data = hepC_train,
  method = "svmRadial",               
  preProcess = c("center", "scale"), # data transformations are estimated
  metric = "ROC",  # area under ROC curve (AUC)       
  trControl = ctrl, # using the trainControl method in the previous step
  tuneLength = 10
)

# Inspect the model parameters and results
hepC_sub_auc

plot(hepC_sub_auc)


## ----model-eval----------------------------------------------------------------------------------------------------
# Use the model developed from the training data to predict Blood or Hepatitis
# within the test data
hepC_test_pred <- predict(hepC_sub_auc, newdata = hepC_test)

# Inspect the model - this only shows which variables have been labelled
hepC_test_pred

# View the confusion matrix of the predicted model, inspecting the Blood and
# Hepatitis predictions
confusionMatrix(hepC_test_pred, hepC_test$Category)



## ----vip-----------------------------------------------------------------------------------------------------------

# Variable importance plot
set.seed(1234)  # for reproducibility

prob_hepC <- function(object, newdata) {
  predict(object, newdata = newdata, type = "prob")[, "Hepatitis"]
}

vip(hepC_sub_auc, method = "permute", nsim = 5, train = hepC_train, 
    target = "Category", metric = "auc", reference_class = "Hepatitis", 
    pred_wrapper = prob_hepC)


