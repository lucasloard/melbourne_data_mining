install.packages('psych')
install.packages("reshape2")
install.packages("ggplot2")
install.packages("broom")
install.packages("ggpubr")
install.packages('xgboost')     # for fitting the xgboost model

install.packages('caret')       # for general data preparation and model fitting
install.packages('MLmetrics')
# Import library
library(MLmetrics)

# install.packages("superml")
library(broom)
library(ggpubr)

# importing required libraries
library(reshape2)
library(ggplot2)
library(psych)
library(superml)
library(xgboost)
library(caret)

print(getwd())
setwd("D:/Money_Making/Data Mining")
print(getwd())

df <- read.csv('melb_data.csv/melb_data.csv')
print(df)
print(is.data.frame(df))
describe(df)
print(factor(df$SellerG))
str(df)

# Change the data types
df$Date <- as.Date(df$Date,"%d/%m/%Y")
str(df$Date)
length(unique(df$Suburb))

## Cheching the unique values of the columns that are categorical
col_names <- colnames(df)
categorical <- list()
non_categorical <- list()
# rm(categoricla)
for (x in col_names){
  if(class(df[[x]])=='character'){
    categorical <- append(categorical,x)
  }
    if(class(df[[x]])=='numeric' || class(df[[x]]) =='integer'){
      non_categorical <- append(non_categorical,x)
      print(x)
    }
  }


print(categorical)

## Dealing with Null Values
# Checking for Null Values in every column
null_columns <- list()
for(x in col_names){
  print(paste(x,class(df[[x]]),sum(is.na(df[[x]])),sep=" "))
  if(sum(is.na(df[[x]]))>0){
    null_columns <- append(null_columns,x)
  }
}
df$Car[is.na(df$Car)]<-mean(df$Car,na.rm=TRUE)
# Creating a copy of df to process the data for Machine Learning applications
df2 <- df

# Dropping the Null columns
# Filling the numeric class columns with mean values of Cars
df2<- df2[ , !(names(df2) %in% null_columns)]
col_name2 <- names(df2)
print(col_name2)

# Checking for Null Values again
for(x in col_name2){
  print(paste(x,class(df2[[x]]),sum(is.na(df2[[x]])),sep=" "))
}
# format(df2$Date,format = '%m')
df2$Year <- format(df2$Date,format = '%Y')
df2$month <- format(df2$Date,format = '%m')
# View(df2)
# Dropping Address
df2 <- df2[,!names(df2) %in% c('Address')]

df2 <- df2[,!names(df2) %in% c('Postcode')]
# df2 <- df2[,!names(df2) %in% c('Lattitude','Longtitude')]
## Extracting Categroical and Numerical columns from New Dataframe
categorical2 <- list()
non_categorical2 <- list()

col_name2 <- names(df2)
print(col_name2)
for (x in col_name2){
  if(class(df2[[x]])=='character'){
    categorical2 <- append(categorical2,x)
  }
  if(class(df2[[x]])=='numeric' || class(df2[[x]]) =='integer'){
    non_categorical2 <- append(non_categorical2,x)
    # print(x)
  }
}




factor_cols <- list()

## Including 34 maximum categorical values
for(x in categorical){
  if(length(unique(df[[x]]))<=34){
    factor_cols <- append(factor_cols,x)
  }
}
# print(factor_cols)

print(length(unique(df2$Postcode)))
# View(df2)
print(non_categorical2)

# View(subset(df2,select = unlist(non_categorical2)))
# View(subset(df2,select = unlist(non_categorical2)))
print(unlist(non_categorical2))

## Dealing with outliers
# Using Box plot to search for outliers
# creating a plot
meltedData <- melt(subset(df2,select = unlist(non_categorical2)))
boxplot(data=meltedData, value~variable)


## Scaling the dataframe for perspective
scaled_numeric <- as.data.frame(scale(subset(df2,select = unlist(non_categorical2))))
# View(scaled_numeric)
meltedData <- melt(scaled_numeric)
boxplot(data=meltedData, value~variable)

# Detect Outliers
# create detect outlier function
detect_outlier <- function(x) {

  # calculate first quantile
  Quantile1 <- quantile(x, probs=.25)

  # calculate third quantile
  Quantile3 <- quantile(x, probs=.75)

  # calculate inter quartile range
  IQR <- Quantile3-Quantile1

  # return true or false
  x > Quantile3 + (IQR*1.5) | x < Quantile1 - (IQR*1.5)
}

remove_outlier <- function(dataframe,
                            columns=names(dataframe)) {

  # for loop to traverse in columns vector
  for (col in columns) {

    # remove observation if it satisfies outlier function
    dataframe <- dataframe[!detect_outlier(dataframe[[col]]), ]
  }

  # return dataframe
  # print("Remove outliers")
  return(dataframe)
}
merged_scaled <- cbind(scaled_numeric,subset(df2,select=unlist(categorical2)))
non_outliers <- remove_outlier(merged_scaled,columns = unlist(non_categorical2))
meltedData2 <- melt(subset(non_outliers,select=unlist(non_categorical2)))

## Removed Outliers
boxplot(data=meltedData2, value~variable)

# View(merged_scaled)

print(non_categorical2)
df3 <- merged_scaled
## Converting all the variables to factors
for(x in categorical2){
  merged_scaled[[x]] <- as.factor(merged_scaled[[x]])
  merged_scaled[[x]] <- as.numeric(factor(merged_scaled[[x]]))

}

for(x in names(merged_scaled)){
  print(paste(x,class(merged_scaled[[x]]),class(merged_scaled[[x]]),sep=" "))
}
# View(merged_scaled)
write.csv(merged_scaled, "D:/Money_Making/Data Mining/merged_scaled.csv", row.names=FALSE)


# creating a train test data
#make this example reproducible
set.seed(123)

#use 70% of dataset as training set and 30% as test set
sample <- sample(c(TRUE, FALSE), nrow(X), replace=TRUE, prob=c(0.7,0.3))
train  <- merged_scaled[sample, ]
test   <- merged_scaled[!sample, ]

X_train <- data.matrix(train[,!names(train) %in% c('Price')])
y_train <- data.matrix(train$Price)

X_test <- data.matrix(test[,!names(test) %in% c('Price')])
y_test <- data.matrix(test$Price)



# Fitting Multiple Linear Regression to the Training set
regressor <- lm(formula = Price ~ .,
               data = train)
# Predicting the Test set results
y_pred_lm = predict(regressor, newdata = test)

print(y_pred_lm)
mse_lm <- mean((y_test - y_pred_lm)^2)
mae_lm <- caret::MAE(y_test, y_pred_lm)
rmse_lm <- caret::RMSE(y_test, y_pred_lm)
mape_lm <- MAPE(y_test, y_pred_lm)
cat("MSE: ", mse_lm, "MAE: ", mae_lm, " RMSE: ", rmse_lm,"MAPE:", mape_lm)
lm_evals <- list(mse_lm,mae_lm,rmse_lm,mape_lm)

#define final training and testing sets
xgb_train <- xgb.DMatrix(data = X_train, label = y_train)
xgb_test <- xgb.DMatrix(data = X_test, label = y_test)

#defining a watchlist
watchlist <- list(train=xgb_train, test=xgb_test)

#fit XGBoost model and display training and testing data at each iteartion
model <- xgb.train(data = xgb_train, max.depth = 3, watchlist=watchlist, nrounds = 100)

#define final model
model_xgboost <- xgboost(data = xgb_train, max.depth = 3, nrounds = 86, verbose = 0)
summary(model_xgboost)

#use model to make predictions on test data
pred_y <- predict(model_xgboost, xgb_test)

# performance metrics on the test data

mse <- mean((y_test - pred_y)^2)
mae <- caret::MAE(y_test, pred_y)
rmse <- caret::RMSE(y_test, pred_y)
mape <- MAPE(y_test, pred_y)
cat("MSE: ", mse, "MAE: ", mae, " RMSE: ", rmse,"MAPE:", mape)
xg_evals <- list(mse,mae,rmse,mape)

# Compute feature importance matrix
importance_matrix <- xgb.importance(colnames(xgb_train), model = model_xgboost)
# importance_matrix
xgb.plot.importance(importance_matrix[1:10,])

# print(length(y_train))
eval_comp <- data.frame(linear_model=unlist(lm_evals),xgb_model=unlist(xg_evals))
View(eval_comp)

plot(df$Distance,df$Price,col = 'Type')