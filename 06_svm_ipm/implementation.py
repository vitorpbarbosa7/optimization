import numpy as np
import pandas as pd
from math import log
from numpy import array as ar
import matplotlib.pyplot as plt
import seaborn as sns

def simple_dataset(size):

    # Two-dimensional dataset
    # Negative examples with mean 0 and sd = 1
    # Positive examples with mean 10 and sd = 1
    
    neg_train = _gen_sample(1,1,size,-1)
    pos_train = _gen_sample(10,1,size,+1)

    train = np.concatenate((neg_train, pos_train), axis = 0)

    neg_test = _gen_sample(1,1,size,-1)
    pos_test = _gen_sample(10,1,size,+1)
    test = np.append(neg_test, pos_test)

    test = np.concatenate((neg_train, pos_train), axis = 0)

    return train,test

def primal_dual_path():

    X_train = train[:,[0,1]]
    y_train = train[:,2].reshape(-1,1)

    npoints = X_train.shape[0]

    # Matrix Q, which is yi*yj*<xi*xj>, a result of the Primal... kkt, and finally a simplification for this 
    # matrix form 
    Q = ( y_train @ y_train.T ) * (X_train @ X_train.T)

    # Hyperparameters to start
    # thinking about the meaning of C (intuition and mathematics understanding according to objective function)
    # https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel
    # answer from deerishi
    C = 1 
    eta = 0.1 # Adapt answers throughout iterations
    b = np.random.uniform(low = -1, high = 1, size = 1) # is this only one number?
    iteration = 1
    threshld = 1E-5

    # vector with ones
    e = np.ones(npoints)

    # Identity matrix
    I = np.identity(npoints)

    # alphas initialized as half of the upper bound C
    Alpha = np.diag([C/2]*npoints)

    # Diagonal matrix Ksi with alpha values as initials
    Ksi = Alpha

    #Diagnonal matrix S (slack variables, according to first equation)
    # f1 = Q*alpha + y*b + Ksi - S - e = 0
    # S is an array of values? slack value for each training example
    S = np.diag(np.asarray((Q @ np.diag(Alpha)) + y_train*b + np.diag(Ksi) - e))
    print(S.shape)

    # Current gap, from the Complementary Slackness of the Primal Dual Path Following
    # Dot product used for this summation
    gap = e @ S * np.diag(Alpha) @ e

    # Initial mu, that is, the term from the barrier from the interior point method using
    mu = gap

    # From the Interior Point Method Framework, we should get closer to the barriers
    reducing_factor = 0.9

    # >> Jacobian 
    jacobian_format = 3*npoints + 1

    # initializing with zero 
    J = np.zeros(shape=(jacobian_format, jacobian_format))

    # Filling it with the initial terms
    # > First row
    a = npoints
    b = npoints+1
    c = 2*npoints+1
    d = 3*npoints+1
    J[0:npoints,0:npoints] = Q
    J[0:npoints,0:npoints+1] = y_train
    J[0:npoints,npoints+1:2*npoints+1] = -I
    J[0:npoints,2*npoints+1:3*npoints+1] = I

    # > Second row
    J[npoints:npoints+1,0:npoints] = -y_train.T
    J[npoints:npoints+1,npoints:3*npoints+1] = 0
    # __viz(J,True)

    # while (gap > threshld):

    # > Third row
    J[npoints+1:2*npoints+1,0:npoints] = S 
    J[npoints+1:2*npoints+1,npoints+1] = 0
    J[npoints+1:2*npoints+1,npoints+1:2*npoints+1] = 0
    J[npoints+1:2*npoints+1,2*npoints+1:3*npoints+1] = 0
    __viz(J,True)

    

    pass


def _gen_sample(mean, sd, size, target:int):

    x1 = np.random.normal(loc = mean, scale = sd, size = size)
    x2 = np.random.normal(loc = mean, scale = sd, size = size)

    target_array = ar([target]*size)

    sample = ar([x1,x2,target_array])    
    
    return sample.T

def __viz(M, annot:bool = False):
    '''
    Visualize Matrix M with Heatpmap
    '''
    sns.heatmap(M, annot = annot)
    plt.show()

if __name__ == '__main__':
    SIZE = 2
    train, test = simple_dataset(SIZE)

    primal_dual_path()