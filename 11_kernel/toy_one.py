import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import array as ar
from sklearn.metrics.pairwise import euclidean_distances


X = pd.read_csv('iris.csv').values
X = X[:,0:2]

xn = X[0]
xm = X[1]

def phi(x:np.array):
    # x single point
    # x ∈ Rd
    # d dimensions
    
    # as x is a single point, x[0] and x[1] are the different dimensions

    lifted_point = [x[0]**2, x[1]**2, np.sqrt(2*x[0]*x[1])]

    return lifted_point

# construct new dataset with Lifting Trick (Elevate to higher dimension)

X_lifted = []
for i in range(X.shape[0]):
    X_lifted.append(phi(X[i]))
X_lifted = ar(X_lifted)

# calculate the dot product:
X_lifted_dot = np.dot(X_lifted, X_lifted.T)

# sns.heatmap(X_lifted_dot)
# plt.show()

# >> With kernel trick 
def homogeneous_polynomial_kernel(xn, xm):
    # xn and xm are different observations, different points belongint to Rd
    # d dimensions (columns, features, axes)
    calculated_kernel_value = (np.dot(xn, xm))**2
    return calculated_kernel_value

#single point for test
print(homogeneous_polynomial_kernel(xn, xm))

# Apply the kernel for all pairwise points, to get the final Gram Matrix, Positive Definite, which allows for convex optimization... 
# and other properties...

# all points
X_matrix_kernel = homogeneous_polynomial_kernel(X, X.T)
# check matrix again
# sns.heatmap(X_matrix_kernel)
# plt.show()

# rbf kernel
def radial_basis_function(xn:np.array, xm:np.array, sigma:float):
    '''''
    In mathematics, a radial function is a real-valued function defined on a Euclidean 
    space Rn whose value at each point depends only on the distance between that point and the origin.
    '''
    
    l2_norms = euclidean_distances(X = xn, Y = xm, squared = True)
    variance = sigma**2
    
    #gamma = 1/variance
    #print(gamma)
    
    k_xn_xm = np.exp(l2_norms/(2*variance))
    
    return k_xn_xm

#test
# radial_basis_function(X[0], X[1], sigma = 1)

# all points
X_matrix_kernel = radial_basis_function(X, X, sigma = 2)
# check matrix again
sns.heatmap(X_matrix_kernel)
plt.show()
