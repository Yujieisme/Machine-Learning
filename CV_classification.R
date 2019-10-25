rm(list=ls())
cat("\014")
library(ISLR)
library(MASS)
library(class)
library(ggplot2)
mydata = College
n = dim(mydata)[1]
p = dim(mydata)[2] - 1
X = data.frame(mydata[,-1])
y = mydata[,1]

random_order = sample(n)

k_fold  = n

fold_size  = floor(n/k_fold)

knn_values = c(seq(1,10,1),seq(11,100,10))
knn_err    = matrix(0, length(knn_values))
log_err    = 0
lda_err    = 0
qda_err    = 0
for (j in 1:k_fold){
  fold_index  =  random_order[(1+ (j-1)*fold_size):(j *fold_size)]
  y.val       =  y[fold_index]
  X.val       =  X[fold_index, ]
  y.train       =  y[-fold_index]
  X.train       =  X[-fold_index, ]

  for (i in 1:length(knn_values)){
    y.val.hat   = knn(X.train, X.val, y.train, k=knn_values[i])
    knn_err[i]  = knn_err[i] + sum(y.val != y.val.hat)
  }
  
  fit  =  glm(Private ~., mydata[-fold_index, ], family="binomial")
  p.val.hat = predict(fit, X.val, "response")
  y.val.hat = rep("No", fold_size)
  y.val.hat[p.val.hat > 0.5] = "Yes"
  log_err  = log_err + sum(y.val != y.val.hat)
  
  fit  =  lda(Private ~., mydata[-fold_index, ])
  y.val.hat = predict(fit, X.val)
  lda_err  = lda_err + sum(y.val != y.val.hat$class)
  
  fit  =  qda(Private ~., mydata[-fold_index, ])
  y.val.hat = predict(fit, X.val)
  qda_err  = qda_err + sum(y.val != y.val.hat$class)
  
  
}

knn_err = knn_err/n
log_err = log_err/n
lda_err = lda_err/n
qda_err = qda_err/n

optimal_k = knn_values[which.min(knn_err)]

knn_err[which.min(knn_err)]

fit  =  glm(Private ~., mydata, family="binomial")
