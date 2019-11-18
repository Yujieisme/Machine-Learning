#input the data
library(MASS)
library(e1071)
library(class)
library(glmnet)
library(Matrix)
library(foreach)
library(randomForest)
df <- read.table("D://statistical learning for machine learning//spambase.data",header = FALSE,sep = ",")
dim(df)

#standardize the dataframe
df[,1:(ncol(df)-1)] = data.frame(scale(df[,1:(ncol(df)-1)]))

#prepare the dataframe
X = df[,-58]
n = dim(X)[1]
p = dim(X)[2]

#create dataframe to store result
lda.train.err = rep(0, 100)
lda.test.err = rep(0, 100)
qda.train.err = rep(0, 100)
qda.test.err = rep(0, 100)
knn.train.err = rep(0, 100)
knn.test.err = rep(0, 100)
lasso.train.err = rep(0, 100)
lasso.test.err = rep(0, 100)
ridge.train.err = rep(0, 100)
ridge.test.err = rep(0, 100)
rf.train.err = rep(0, 100)
rf.test.err = rep(0, 100)

#cut the train/test set
n1 = round(n/2)

#generate noise for qda 
noise = matrix(rnorm(n=131100, mean=0, sd=0.01),nrow = 2300, ncol = 57)

#tune K using 10-fold cross validation
knn.cross <- tune.knn(x = as.matrix(X), y = as.factor(df$V58), k = 1:100,tunecontrol=tune.control(sampling = "cross"), cross=10)
summary(knn.cross)
plot(knn.cross)


#prepare data for lasso and ridge
X = model.matrix(V58~.,df)[,-1]
y = df$V58
grid = 10^seq(10,-2,length=100)


for (i in 1:100){
  train = sample(n, n1)
  df.train=df[train,]
  df.test=df[-train,]
  X.train = df.train[,-58]
  y.train = df.train$V58
  X.test = df.test[,-58]
  y.test = df.test$V58
  #fit the lda model
  lda.fit=lda(V58~.,data=df.train)
  lda.train.pred=predict(lda.fit,X.train)
  lda.test.pred=predict(lda.fit,X.test)
  lda.train.err[i] = mean(y.train != lda.train.pred$class)
  lda.test.err[i] = mean(y.test != lda.test.pred$class)
  #fit the qda model
  df.train.noise = data.frame(data.matrix(df.train[,-58])+noise, V58.q = df.train[,58])
  qda.fit=qda(V58.q~.,data=df.train.noise)
  qda.train.pred=predict(qda.fit,X.train)
  qda.test.pred=predict(qda.fit,X.test)
  qda.train.err[i] = mean(y.train != qda.train.pred$class)
  qda.test.err[i] = mean(y.test != qda.test.pred$class)
  #fit the knn model
  knn.train.pred <- knn(train = X.train, test = X.train, cl = y.train, k=3)
  knn.test.pred <- knn(train = X.train, test = X.test, cl = y.train, k=3)
  knn.train.err[i] = sum(knn.train.pred != y.train)/n1
  knn.test.err[i] = sum(knn.test.pred != y.test)/(n - n1)
  #fit lasso regression
  cv.lasso <- cv.glmnet(X[train,],y[train],alpha=1,family = "binomial",intercept = TRUE, standardize = FALSE,  nfolds = 10, type.measure="class")
  best.lambda.lasso = cv.lasso$lambda.min
  lasso.fit <- glmnet((model.matrix(V58~.,df)[,-1][train,]),y[train],alpha=1, family = "binomial", lambda = best.lambda.lasso)
  lasso.train.prob = predict(lasso.fit,s=best.lambda.lasso,newx=model.matrix(V58~.,df)[,-1][train,])
  lasso.train.pred = df.train$V58
  lasso.train.pred[lasso.train.prob>0.5] = 1
  lasso.train.pred[lasso.train.prob<0.5] = 0
  lasso.test.prob = predict(lasso.fit,s=best.lambda.lasso,newx=model.matrix(V58~.,df)[,-1][test,])
  lasso.test.pred = df.test$V58
  lasso.test.pred[lasso.test.prob>0.5] = 1
  lasso.test.pred[lasso.test.prob<0.5] = 0
  lasso.train.err[i] = mean(y.train != lasso.train.pred)
  lasso.test.err[i] = mean(y.test != lasso.test.pred)
  #fit ridge regression
  cv.ridge <- cv.glmnet(X[train,],y[train],alpha=0,family = "binomial",intercept = TRUE, standardize = FALSE,  nfolds = 10, type.measure="class")
  best.lambda.ridge = cv.ridge$lambda.min
  ridge.fit <- glmnet((model.matrix(V58~.,df)[,-1][train,]),y[train],alpha=0, family = "binomial", lambda = best.lambda.ridge)
  ridge.train.prob = predict(ridge.fit,s=best.lambda.ridge,newx=model.matrix(V58~.,df)[,-1][train,])
  ridge.train.pred = df.train$V58
  ridge.train.pred[ridge.train.prob>0.5] = 1
  ridge.train.pred[ridge.train.prob<0.5] = 0
  ridge.test.prob = predict(ridge.fit,s=best.lambda.ridge,newx=model.matrix(V58~.,df)[,-1][test,])
  ridge.test.pred = df.test$V58
  ridge.test.pred[ridge.test.prob>0.5] = 1
  ridge.test.pred[ridge.test.prob<0.5] = 0
  ridge.train.err[i] = mean(y.train != ridge.train.pred)
  ridge.test.err[i] = mean(y.test != ridge.test.pred)
  #fit random forest
  rf.fit = randomForest(V58~.,data=df,subset=train,mtry=sqrt(p),ntree=300,importance=TRUE)
  rf.train.prod = predict(rf.fit,newdata=df.train)
  rf.test.prod = predict(rf.fit,newdata=df.test)
  rf.train.pred = df.train$V58
  rf.train.pred[rf.train.prod>0.5] = 1
  rf.train.pred[rf.train.prod<0.5] = 0
  rf.test.pred[rf.test.prod>0.5] = 1
  rf.test.pred[rf.test.prod<0.5] = 0
  rf.train.err[i] = mean(y.train != rf.train.pred)
  rf.test.err[i] = mean(y.test != rf.test.pred)  
}

train.err = data.frame(lda.train.err, qda.train.err, knn.train.err, lasso.train.err, ridge.train.err, rf.train.err)
test.err = data.frame(lda.test.err, qda.test.err, knn.test.err, lasso.test.err, ridge.test.err, rf.test.err)
par(mfrow=c(2,1))
boxplot(train.err, main = "100 repeated training errors for different models")
boxplot(test.err, main = "100 repeated test errors for different models")





