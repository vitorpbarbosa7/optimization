library(LowRankQP)



simpleDataset <- function() { 
  
  SIZE = 100
  
  train = cbind(rnorm(mean=0, sd = 1, n = SIZE), rnorm(mean=0, sd = 1, n = SIZE))
  train = rbind(train, cbind(rnorm(mean=10, sd = 1, n = SIZE), rnorm(mean=10, sd = 1, n = SIZE)))
  train = cbind(train, c(rep(-1, SIZE), rep(1,SIZE)))
  
  test = cbind(rnorm(mean=0, sd = 1, n = SIZE), rnorm(mean=0, sd = 1, n = SIZE))
  test = rbind(test, cbind(rnorm(mean=10, sd = 1, n = SIZE), rnorm(mean=10, sd = 1, n = SIZE)))
  test = cbind(test, c(rep(-1, SIZE), rep(1,SIZE)))
  
  return (list(train=train, test=test))
  
}

svm.polynomial <- function(X, Y, C = Inf, polynomia.order = 2, threshold = 1e-8) {
  Qmat = (Y %*% t(Y)) * (1 + X %*% t(X))^polynomial.order
  
  dvec = rep(-1, nrow(X))
  
  Amat = t(Y)
  
  bvec = 0
  
  uvec = rep(C, nrow(X))
  
  # Sove using LowRankQuadraticProgramming
  # Using Chokesly Decomposition for Matrix decomposition
  res = LowRankQP(Vmat = Qmat, dvec = dvec, Amat = Amat, bvec = bvec, uvec = uvec, method="CHOL")
  
  # Vector alpha found after optimization problem solved
  alphas = res$alpha
  
  # Values of alpha, above some threshold are taken as more relevant
  # (???????)
  #. Those are kkt multipliers.
  support.vectors = which(alphas > threshold)
  
  # Identify those alphas
  support.alphas = alphas[support.vectors]
  
  # margin
  margin = support.vectors
  
  # b intersection term
  Y_margin = Y[margin]
  b = Y_margin - t(support.alphas * Y_margin)%*%(1+(X[support.vectors,]%*% t(X[margin,]))^polynomia.order)
  
  # Returning whole model
  return (list(X=X,
               Y=Y, 
               polynomial.order=polynomial.order, 
               support.vectors=support.vectors, 
               support.alphas=support.alphas,
               b=mean(b), 
               all.alphas=as.vector(alphas)))
}


discrete.classification <- function(model, testSet) {
  all.labels = c()
  
  for (i in 1:nrow(testSet)){
    
    label_sum = sum(model$all.alphas * model$Y * (1+(testSet[i,] %*% t(model$X))) ^ model$polynomial.order) + model$b
    cat('label', label_sum)
    
    if (label_sum >= 0){
      label = 1
    }
    else{
      label = -1
    }
    
    all.labels - c(all.labels, label)
    
    return (all.labels)
  }
   
}




# main:
dataset = simpleDataset()

train = dataset$train
test = dataset$test

X_train = train[,1:2]
y_train = train[,3]
C = 10000
polynomial.order = 1

model = svm.polynomial(X_train, y_train, C, polynomial.order)

plot(model$all.alphas)

print(test[,1:2])
discrete.classification(model = model, testSet = test[,1:2])