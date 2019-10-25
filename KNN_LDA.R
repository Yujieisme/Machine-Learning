rm(list = ls())    #delete objects
cat("\014")
#R packages are a collection of R functions, complied code and sample data.
library(class) # KNN 
library(ggplot2) # A library to generate high-quality graphs
library(dplyr) # A library for data manipulation
library(gridExtra) # Provides a number of user-level functions to work with "grid" graphics, notably to arrange multiple grid-based plots on a page, and draw tables
#set.seed(1)
library(MASS)


# generate data
n          =     1000
p          =     2
X          =     matrix(ncol = p, nrow = n)
p1         =     0.5
rho1       =     0.9
rho0       =     -0.9
o1         =     0.5
o2         =     0.5
C1         =     o1^2 * matrix(c(1, rho1, rho1, 1), nrow = 2)
C0         =     o2^2 * matrix(c(1, rho0, rho0, 1), nrow = 2)
mu1        =     c(-5,1)/sqrt(2)
mu0        =     c(3,-1)/sqrt(2)
o          =     0.5
#mvrnorm(n = 1, mu, Sigma, tol = 1e-6, empirical = FALSE, EISPACK = FALSE)

# Data generating process
 y          =     as.factor(rbinom(n, 1, p1)) # Conceptually, factors are variables in R which take on a limited number of different values; such variables are often refered to as categorical variables.
# for (i in 1:n){
#   mu     =     (y[i]==1)*mu1 + (y[i]==0)*mu0
#   C      =     (y[i]==1)*C1 + (y[i]==0)*C0
#   X[i,]  =     mvrnorm(1, mu, C) 
# }

# Data generating process

# 
k  =  0;
for (i in 1:n){
  if (y[i] == 1){
    X[i,]  =     rnorm(p, mean = 0, sd = o) 
  } else {
    k      =     k +1
    X[i,]  =     rnorm(p, mean = 0, sd = o) + 5*o*c(cos(2*pi * k / (n/10)) , sin(2*pi * k / (n/10)))
  }
}


# separate the features corresponding to 1 and 0
n1        =   sum(y==1)
n0        =   sum(y==0)
X1        =   X[y==1,]
X0        =   X[y==0,]
mu1.hat   =   colMeans(X1)
mu0.hat   =   colMeans(X0)
p1.hat    =   n1/n
p0.hat    =   1 - p1.hat

# below we want to subtract the row vector mu1.hat from each row
#X1        =   X1 - mu1.hat  <----THIS WAS WRONG - SEE BELOW FOR CORRECTION
X1        =   X1 -t(matrix(rep(mu1.hat,n1), ncol = n1))
#X0        =   X0 - mu0.hat  <----THIS WAS WRONG - SEE BELOW FOR CORRECTION
X0        =   X0 -t(matrix(rep(mu0.hat,n0), ncol = n0))



C.hat     =   cov(rbind(X1,X0))

beta.hat  =  solve(C.hat) %*% (mu1.hat-mu0.hat)
beta0.hat =  -0.5* mu1.hat %*% solve(C.hat) %*% mu1.hat + 0.5* mu0.hat %*% solve(C.hat) %*% mu0.hat + log(p1.hat/p0.hat)

 X.grid     =     expand.grid(x=seq(from = min(X[,1]), to =  max(X[,1]), length.out = 100), 
                              y=seq(from = min(X[,2]), to =  max(X[,2]), length.out = 100))

y.hat.grid =     as.matrix(X.grid) %*% beta.hat + as.vector(beta0.hat)
y.hat.grid =     (y.hat.grid > 0) * 1;


dataff     =     data.frame( X.grid, y.hat.grid)
dataf      =     data.frame(y, X)

p1         =     ggplot(dataf, aes(x=X1, y=X2, colour=y))+geom_point()+ggtitle("train data")
p3         =     ggplot(dataff, aes(x=x, y=y, colour=y.hat.grid))+geom_point()+ggtitle("predicted label")
grid.arrange(p1,  p3, nrow =1) 













 
 k =  3
 
 
 X.grid     =     expand.grid(x=seq(from = min(X[,1]), to =  max(X[,1]), length.out = 100), 
                              y=seq(from = min(X[,2]), to =  max(X[,2]), length.out = 100))
 y.hat.grid =     knn(X, X.grid, y, k = k, prob = TRUE)
 
 prob.grid  =     attr(y.hat.grid, "prob")
 prob.grid  =     ifelse(y.hat.grid=="1", prob.grid, 1-prob.grid)
 dataff     =     data.frame(prob.grid, X.grid, y.hat.grid)
 dataf      =     data.frame(y, X)
 
 p1         =     ggplot(dataf, aes(x=X1, y=X2, colour=y))+geom_point()+ggtitle("train data")
 p2         =     ggplot(dataff, aes(x=x, y=y, colour=prob.grid))+geom_point()+ggtitle("predicted probability")
 p3         =     ggplot(dataff, aes(x=x, y=y, colour=y.hat.grid))+geom_point()+ggtitle("predicted label")
 grid.arrange(p1, p2, p3, nrow =1) 



