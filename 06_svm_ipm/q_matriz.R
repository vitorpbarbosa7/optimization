
x1 = c(1,2,3)
x2 = c(2,3,4)
x3 = c(2,3,5)
X = rbind(x1,x2,x3)

mat = X%*%X

cat(dim(mat))
