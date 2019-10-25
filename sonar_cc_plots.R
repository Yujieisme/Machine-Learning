rm(list = ls())    #delete objects
cat("\014")
library(class)
library(ggplot2)
library(dplyr)
library(glmnet)
#library(alocv)
library(rmutil)
library(tictoc)
library(latex2exp)

setwd("~/Documents/Teaching/Spring 2018 - STA 9890/Codes/sonar")
# data is from http://math.furman.edu/~dcs/courses/math47/R/library/mlbench/html/Sonar.html
# Gorman, R. P., and Sejnowski, T. J. (1988). "Analysis of Hidden Units in a Layered Network Trained to Classify Sonar Targets" in Neural Networks, Vol. 1, pp. 75-89.
# 

sonar	        =	   read.csv("sonar.all-data",header=FALSE)
X             =    model.matrix(V61~., sonar)[, -1]
y             =    sonar$V61
y             =    (y=='R')*1
n             =    dim(X)[1] # sample size
p             =    dim(X)[2] # number of predictors/features

nfold         =    n;


m             =    30
lasso.cv      =    cv.glmnet(X, y, family = "binomial", alpha = 1,  intercept = TRUE, standardize = FALSE,  nfolds = nfold, type.measure="class")
lam.lasso     =    exp(seq(log(max(lasso.cv$lambda)),log(0.00001), (log(0.00001) - log(max(lasso.cv$lambda)))/(m-1)))



ptm           =     proc.time()
lasso.cv      =     cv.glmnet(X, y, lambda = lam.lasso, family = "binomial", alpha = 1,  intercept = TRUE, standardize = FALSE,  nfolds = nfold, type.measure="class")
ptm           =     proc.time() - ptm
time.lasso.cv =     ptm["elapsed"] 

ptm           =     proc.time()
lasso.fit     =     glmnet(X, y, lambda = lasso.cv$lambda, family = "binomial", alpha = 1,  intercept = TRUE, standardize = FALSE)
ptm           =     proc.time() - ptm
time.lasso.fit=     ptm["elapsed"] 

lasso.fit.0   =    glmnet(X, y, lambda = 0, family = "binomial", alpha = 1,  intercept = TRUE, standardize = FALSE)
n.lambdas     =    dim(lasso.fit$beta)[2]

lasso.beta.ratio    =    rep(0, n.lambdas)
for (i in 1:n.lambdas) {
  lasso.beta.ratio[i]   =   sum(abs(lasso.fit$beta[,i]))/sum(abs(lasso.fit.0$beta))
}

ridge.cv      =    cv.glmnet(X, y, family = "binomial", alpha = 0,  intercept = TRUE, standardize = FALSE,  nfolds = nfold, type.measure="class")
lam.ridge     =    exp(seq(log(max(ridge.cv$lambda)),log(0.00001), -(log(max(ridge.cv$lambda))-log(0.00001))/(m-1)))


ptm                 =     proc.time()
ridge.cv            =     cv.glmnet(X, y, lambda = lam.ridge, family = "binomial", alpha = 0,  intercept = TRUE, standardize = FALSE,  nfolds = nfold, type.measure="class")
ptm                 =     proc.time() - ptm
time.ridge.cv       =     ptm["elapsed"] 

ptm                 =     proc.time()
ridge.fit           =     glmnet(X, y, lambda = ridge.cv$lambda, family = "binomial", alpha = 0,  intercept = TRUE, standardize = FALSE)
ptm                 =     proc.time() - ptm
time.ridge.fit      =     ptm["elapsed"] 


ridge.fit.0         =    glmnet(X, y, lambda = 0, family = "binomial", alpha = 0,  intercept = TRUE, standardize = FALSE)


n.lambdas     =    dim(ridge.fit$beta)[2]
ridge.beta.ratio    =    rep(0, n.lambdas)
for (i in 1:n.lambdas) {
  ridge.beta.ratio[i]   =   sqrt(sum((ridge.fit$beta[,i])^2)/sum((ridge.fit.0$beta)^2))
}

eror           =     data.frame(c(rep("lasso", length(lasso.beta.ratio)),  rep("ridge", length(ridge.beta.ratio)) ), 
                                c(lasso.beta.ratio, ridge.beta.ratio) ,
                                c(lasso.cv$cvm, ridge.cv$cvm),
                                c(lasso.cv$cvsd, ridge.cv$cvsd))
colnames(eror) =     c("method", "ratio", "cv", "sd")

eror.plot      =     ggplot(eror, aes(x=ratio, y = cv, color=method)) +   geom_line(size=1) 
eror.plot      =     eror.plot  + scale_x_log10()#(breaks = c(seq(0.1,2.4,0.2)))   
eror.plot      =     eror.plot  + theme(legend.text = element_text(colour="black", size=16, face="bold", family = "Courier")) 
eror.plot      =     eror.plot  + geom_pointrange(aes(ymin=cv-sd, ymax=cv+sd),  size=0.8,  shape=15)
eror.plot      =     eror.plot  + theme(legend.title=element_blank()) 
eror.plot      =     eror.plot  + scale_color_discrete(breaks=c("lasso", "ridge"))
eror.plot      =     eror.plot  + theme(axis.title.x = element_text(size=24),
                                        axis.text.x  = element_text(angle=0, vjust=0.5, size=14),
                                        axis.text.y  = element_text(angle=0, vjust=0.5, size=14)) 
#eror.plot      =     eror.plot  + theme(axis.title.y = element_text(size=16, face="bold", family = "Courier")) 
#eror.plot      =     eror.plot  + xlab( expression(paste( lambda))) + ylab("")
eror.plot      =     eror.plot  + theme(plot.title = element_text(hjust = 0.5, vjust = -10, size=20, family = "Courier"))
#eror.plot      =     eror.plot  + ggtitle(TeX(sprintf("$n$=%s,$p$=%s,$t_{LO}$=%s,$t_{ALO}$=%0.3f,$t_{FIT}$=%.3f",n,p,time.lo,time.alo,time.fit))) 
eror.plot      =     eror.plot  + ggtitle((sprintf("lasso.cv:%0.3f(sec), lasso.fit:%0.3f(sec) \n ridge.cv:%0.3f(sec), ridge.fit:%0.3f(sec)",time.lasso.cv,time.lasso.fit,time.ridge.cv,time.ridge.fit))) 

eror.plot

