rm(list = ls()) #delete object
cat("\014")
n         =     5   
p         =     2
i         =     1:n
X         =     matrix(0, nrow = n, ncol = p)
X[ ,1]    =     cos((4*i-3)/10 * pi)
X[ ,2]    =     sin((4*i-3)/10 * pi)
y         =     c(1,1,1,-1,-1)
beta0     =     matrix(0, nrow = 1, ncol = 3)
beta      =     matrix(0, nrow = 2, ncol = 3)

beta0[1]  =     -0.5
beta[,1]  =     c(0,1)

t         =     1
x_1       =     seq(-1,1,0.1)
x_2       =     -beta0[t]/beta[2,t] - beta[1,t]/beta[2,t] * x_1

plot(X, pch = 15, col = (y+3) )# plot the data
lines(x_1, x_2, col="black")

rho    =    0.5

DH     =    (X %*% beta[ ,t] + beta0[t]) / sqrt(sum(beta[ ,t]^2))*y
M      =    which(DH < 0)
indx   =    M[1]

beta[, t+1]    =    beta[ ,t] + rho * y[indx] * t(X[indx, ])
beta0[ ,t+1]   =    beta0[, t] + rho * y[indx]

t         =     t+1
x_1       =     seq(-1,1,0.1)
x_2       =     -beta0[t]/beta[2,t] - beta[1,t]/beta[2,t] * x_1

plot(X, pch = 15, col = (y+3) )
lines(x_1, x_2, col="black")

DH     =    (X %*% beta[ ,t] + beta0[t]) / sqrt(sum(beta[ ,t]^2))*y
M      =    which(DH < 0)
indx   =    M[1]

beta[, t+1]    =    beta[ ,t] + rho * y[indx] * t(X[indx, ])
beta0[ ,t+1]   =    beta0[, t] + rho * y[indx]

t         =     t+1
x_1       =     seq(-1,1,0.1)
x_2       =     -beta0[t]/beta[2,t] - beta[1,t]/beta[2,t] * x_1

plot(X, pch = 15, col = (y+3) )
lines(x_1, x_2, col="black")

DH     =    (X %*% beta[ ,t] + beta0[t]) / sqrt(sum(beta[ ,t]^2))*y
M      =    which(DH < 0)
indx   =    M[1]

