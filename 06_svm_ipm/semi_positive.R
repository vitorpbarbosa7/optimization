
size = 10
Q = diag(size)

alphas = matrix(rnorm(mean=0,sd=1,n=size),ncol = 1)

res = t(alphas)%*%Q%*%alphas     

image(Q)


# GRAM MATRIX 
# something about: https://www.youtube.com/watch?v=3-Ds_K3p_Z8