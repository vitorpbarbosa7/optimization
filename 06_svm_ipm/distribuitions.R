half_size = 2
X = cbind(rnorm(mean=-1,sd=0.5,n=half_size), rnorm(mean=-1,sd=0.5, half_size))
X = rbind(X, cbind(rnorm(mean=1,sd=0.5,half_size), rnorm(mean=1,sd=0.5, half_size)))

size = dim(X)[1]

# plot(X, pch=20, cex=3)

y = c(rep(+1,half_size), rep(-1,half_size))

plot(X, pch=20, cex=3, col=y+1)
plot(X, pch=20, cex=3, col=y+2)

#y summation, i, j
y_matrix = y%*%t(y)
dim(y_matrix)

# Medida da similaridade de um vetor com o outro
X_matrix = X%*%t(X)
dim(X_matrix)

#image(X_matrix)

Q = y_matrix * t(X_matrix)
dim(Q)

image(Q)

# Valores característicos da matriz Q
# eigen(característica)
# eigenvalue : O quanto ocorre de "encolhimento" "esticamento" da transformação linear resultante daquele vetor
# eigenvector: Para onde é esticada a transformação linear
eig = eigen(Q)

# Decomposicao em eigenvalues e eigenvectors
eig$values  

# Decomposicao de cholesky
# Gerando aproximacao para tornar o problema convexo e possivel de resolver
Q_approx = chol(Q + 1e-3*diag(size))
Q_approx