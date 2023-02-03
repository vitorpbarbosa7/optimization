import numpy as np
from numpy import array as ar
from scipy.linalg import null_space
from numpy.linalg import eig, norm

Al = ar([[1,1],[0,1],[-1,1]])
A = ar([[2,0],[0,3]])

carac = eig(A)

eigvals = carac[0]
eigvectors = carac[1]

v1 = eigvectors[0]
v2 = eigvectors[1]

sig1 = np.sqrt(eigvals[0])
sig2 = np.sqrt(eigvals[1])

null = null_space(Al.T)/norm(null_space(Al.T))
null_ = null[:,0]
U = ar([(1/sig1)*Al@v1, (1/sig2)*Al@v2, null_])

print(U)
