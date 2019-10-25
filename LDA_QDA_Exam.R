rm(list = ls())    #delete objects
cat("\014")
set.seed(1)
#R packages are a collection of R functions, complied code and sample data.
library(nnet) 
library(ggplot2) # A library to generate high-quality graphs
library(dplyr) # A library for data manipulation
library(gridExtra) # Provides a number of user-level functions to work with "grid" graphics, notably to arrange multiple grid-based plots on a page, and draw tables
#set.seed(1)
library(MASS)


# generate data
n          =     1000
p          =     2
X          =     matrix(ncol = p, nrow = n)
J          =     3;
pii        =     c(0.6,0.3,0.1)
rho        =     c(0.25, 0.5, 0.75)
o          =     0.5
C1         =     o^2 * matrix(c(1, rho[1], rho[1], 1), nrow = 2)
C2         =     o^2 * matrix(c(1, rho[2], rho[2], 1), nrow = 2)
C3         =     o^2 * matrix(c(1, rho[3], rho[3], 1), nrow = 2)

mu1        =     c(0,1)
mu2        =     c(1,0)
mu3        =     c(-1,1)

#mvrnorm(n = 1, mu, Sigma, tol = 1e-6, empirical = FALSE, EISPACK = FALSE)

# Data generating process
y                    =     rep(0,n)
unif                 =     runif(n)
class1ID             =     unif < pii[1]    
y[class1ID]          =     1; 
class2ID             =     (unif < (pii[2]+pii[1]) ) & (pii[1] < unif)
y[class2ID]          =     2;
class3ID             =     ((pii[2]+pii[1]) < unif)
y[class3ID]          =     3;
y                    =     as.factor(y) 

for (i in 1:n){
   mu     =     (y[i]==1)*mu1 + (y[i]==2)*mu2 + (y[i]==3)*mu3
   C      =     (y[i]==1)*C1  + (y[i]==2)*C2  + (y[i]==3)*C3
   X[i,]  =     mvrnorm(1, mu, C) 
 }

# Optimal Bayes Classifier and QDA

X.grid          =     expand.grid(x=seq(from = min(X[,1]), to =  max(X[,1]), length.out = 100),                              y=seq(from = min(X[,2]), to =  max(X[,2]), length.out = 100))
y.hat.grid.BC   =     rep(0, nrow(X.grid))
y.hat.grid.QDA  =     rep(0, nrow(X.grid))

# separate the features corresponding to 1 and 0
pii.hat         =   c(sum(y==1)/n, sum(y==2)/n, sum(y==3)/n)
X1              =   X[y==1,]
X2              =   X[y==2,]
X3              =   X[y==3,]

mu1.hat         =   colMeans(X1)
mu2.hat         =   colMeans(X2)
mu3.hat         =   colMeans(X3)
C1.hat          =   cov(X1)
C2.hat          =   cov(X2)
C3.hat          =   cov(X3)


for (i in 1:nrow(X.grid)){
  delta1                =    -0.5 * t(mu1 - t(X.grid[i, ])) %*% solve(C1) %*% (mu1 - t(X.grid[i, ])) + log(pii[1]) -0.5 * log(det(C1))
  delta2                =    -0.5 * t(mu2 - t(X.grid[i, ])) %*% solve(C2) %*% (mu2 - t(X.grid[i, ])) + log(pii[2]) -0.5 * log(det(C2))
  delta3                =    -0.5 * t(mu3 - t(X.grid[i, ])) %*% solve(C3) %*% (mu3 - t(X.grid[i, ])) + log(pii[3]) -0.5 * log(det(C3))
  y.hat.grid.BC[i]      =    which.is.max(c(delta1, delta2, delta3))
  
  delta1.hat            =    -0.5 * t(mu1.hat - t(X.grid[i, ])) %*% solve(C1.hat) %*% (mu1.hat - t(X.grid[i, ])) + log(pii.hat[1]) -0.5 * log(det(C1.hat))
  delta2.hat            =    -0.5 * t(mu2.hat - t(X.grid[i, ])) %*% solve(C2.hat) %*% (mu2.hat - t(X.grid[i, ])) + log(pii.hat[2]) -0.5 * log(det(C2.hat))
  delta3.hat            =    -0.5 * t(mu3.hat - t(X.grid[i, ])) %*% solve(C3.hat) %*% (mu3.hat - t(X.grid[i, ])) + log(pii.hat[3]) -0.5 * log(det(C3.hat))
  y.hat.grid.QDA[i]     =    which.is.max(c(delta1.hat, delta2.hat, delta3.hat))
}

y.hat.grid.BC     =   as.factor(y.hat.grid.BC)
y.hat.grid.QDA    =   as.factor(y.hat.grid.QDA)


dataf.QDA     =     data.frame(X.grid, y.hat.grid.QDA)
dataf.BC      =     data.frame(X.grid, y.hat.grid.BC)
dataf         =     data.frame(y, X)

ptrain       =     ggplot(dataf, aes(x=X1, y=X2, colour=y))+geom_point()+ggtitle("train data")
pBC          =     ggplot(dataf.BC, aes(x=x, y=y, colour=y.hat.grid.BC))+geom_point()+ggtitle("Bayes Clasifier")
pQDA         =     ggplot(dataf.QDA, aes(x=x, y=y, colour=y.hat.grid.QDA))+geom_point()+ggtitle("QDA")

grid.arrange(ptrain, pBC, pQDA, nrow =1) 



