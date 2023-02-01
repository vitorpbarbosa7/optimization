# reference: https://www.youtube.com/playlist?list=PLKWX1jIoUZaVpVhMfevAE7iYNcDHPEJI_
import numpy as np
import pandas as pd
import time
from math import log
from numpy import array as ar
from numpy.linalg import solve
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *

from sklearn.metrics import accuracy_score

def classification(X_train, y_train, alphas, b, X_test, y_test, threshold:float = 1E-5):

    all_labels = None

    alphas.setflags(write = True)
    alphas[alphas < threshold] = 0

    # alpha_y = alphas * np.ravel(y_train)
    # x_multi = X_test[1] @ X_train.T
    # print(alpha_y.shape)
    # print(x_multi.shape)
    # print(sum(alpha_y*x_multi))

    all_labels = []
    y_hats = []
    for i in range(len(X_test)):
        # label = sum((alphas @ np.ravel(y_train)) * (X_test[i] @ X_train≠–=.T)) + b
        label = sum((alphas * np.ravel(y_train)) * (X_test[i] @ X_train.T)) + b
        # print(f'Calculated label: {label}')

        if label >= 0:
            label = 1
        else:
            label = -1
        # print(f'Classificated label: {label}')
        expected_label = list(y_test[i])

        all_labels.append([expected_label,label])
        y_hats.append(label)

    return all_labels, y_hats



def simple_dataset(size):

    # Two-dimensional dataset
    # Negative examples with mean 0 and sd = 1
    # Positive examples with mean 10 and sd = 1
    
    neg_train = _gen_sample(1,1,size,-1)
    pos_train = _gen_sample(10,1,size,+1)

    train = np.concatenate((neg_train, pos_train), axis = 0)

    neg_test = _gen_sample(1,1,size,-1)
    pos_test = _gen_sample(10,1,size,+1)

    test = np.concatenate((neg_test, pos_test), axis = 0)

    return train,test

def primal_dual_path(X_train, y_train):

    npoints = X_train.shape[0]

    # Matrix Q, which is yi*yj*<xi*xj>, a result of the Primal... kkt, and finally a simplification for this 
    # matrix form 
    Q = ( y_train @ y_train.T ) * (X_train @ X_train.T)
    # print(Q)
    # print(Q.shape)

    # Hyperparameters to start
    # thinking about the meaning of C (intuition and mathematics understanding according to objective function)
    # https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel
    # answer from deerishi
    C = 2
    eta = 0.03 # Adapt answers throughout iterations
    b = np.random.uniform(low = -1, high = 1, size = 1) # is this only one number?????
    iteration = 1
    threshold = 1E-5

    # vector with ones
    e = np.ones(npoints).reshape(-1,1)
    print(f'e.T: \n {e.T.shape}')
    print(f'e: \n {e.shape}')

    # Identity matrix
    I = np.identity(npoints)

    # alphas initialized with some value between the lower bound 0 and the the upper bound C
    Alpha = np.diag([C/2]*npoints)
    print(f'Initial Alpha: \n {Alpha}')
    print(Alpha.shape)

    # Diagonal matrix Ksi with alpha values as initials
    Ksi = Alpha

    #Diagnonal matrix S (slack variables, according to first equation)
    # f1 = Q*alpha + y*b + Ksi - S - e = 0
    # S is an array of values? slack value for each training example
    S = np.diag(np.diag((Q @ np.diag(Alpha)) + y_train*b + np.diag(Ksi) - e))
    print(f'Initial S: \n {S}')
    print(S.shape)

    # Current gap, from the Complementary Slackness of the Primal Dual Path Following
    # Dot product used for this summation
    gap = np.diag(S) @ np.diag(Alpha)
    # print(f'Initial gap: {gap}')

    # Initial mu, that is, the term from the barrier from the interior point method using
    mu = gap / 2
    # print(f'Initial mu: {mu}')

    # >> Jacobian 
    jacobian_format = 3*npoints + 1

    # initializing with zero 
    J = np.zeros(shape=(jacobian_format, jacobian_format))

    # indices for filling:
    a_block = npoints
    b_block = npoints+1
    c_block = 2*npoints+1
    d_block = 3*npoints+1
    
    # > First row
    J[0:a_block,0:a_block] = Q
    J[0:a_block,a_block:b_block] = y_train
    J[0:a_block,b_block:c_block] = -I
    J[0:a_block,c_block:d_block] = I

    # > Second row
    J[a_block:b_block,0:a_block] = -y_train.T
    J[a_block:b_block,a_block:b_block] = 0
    J[a_block:b_block,b_block:c_block] = 0
    J[a_block:b_block,c_block:d_block] = 0 # verbose
    # __viz(J,True)

    # >> Right hand side of the System of Equations
    # Vector of B, of right side linear system (**Equations**) (Jacobian * derivatives = **Equations**):
    B = np.zeros(jacobian_format)
    # B = np.zeros(2*npoints+1)

	# From this point, the jacobian matrix has terms which will modified using the interior point method with the barrier
    gaps = []
    while (gap > threshold):
    # for i in range(100): # test iterations before solve whole problem

        # > Third row
        J[b_block:c_block,0:a_block] = S
        J[b_block:c_block,a_block:b_block] = 0
        J[b_block:c_block,b_block:c_block] = Alpha
        J[b_block:c_block,c_block:d_block] = 0

        # Fourth row
        J[c_block:d_block,0:a_block] = -Ksi
        J[c_block:d_block,a_block:b_block] = 0
        J[c_block:d_block,b_block:c_block] = 0
        J[c_block:d_block,c_block:d_block] = (C-Alpha)
        # __viz(J,True)

        # >> Right hand side of the System of Equations
        # First function (f1)
        f1 = np.diag((-Q)@np.diag(Alpha) - y_train*b - np.diag(Ksi) + np.diag(S) + e)
        # print(f'f1 : {f1}')

        # Second function (f2)
        f2 = np.diag(Alpha) @ y_train
        # print(f'f2 : {f2}')

        # Third function (f3)
        f3 = -np.diag(((S) @ Alpha)) + mu
        # print(f'f3 : {f3}')

        # Fourth function (f4)
        f4 = -np.diag((C - Alpha)) @ np.diag(Ksi) + mu
        # print(f'f4 : {f4}')

        # Filling B
        B[0:a_block] = f1
        B[a_block:b_block] = f2 # scalar
        B[b_block:c_block] = f3
        B[c_block:d_block] = f4

        # Once we have a system with real numbers on it, we can solve a linear system
        derivatives = solve(J,B)
        
        # individual derivatives of each term from the original Primal Dual Problem
        d_alpha = derivatives[0:a_block]
        d_b = derivatives[a_block:b_block]
        d_slack = derivatives[b_block:c_block]
        d_ksi = derivatives[c_block:d_block]

        # Update variables, according to where the mimization points to (Jacobian values for the gradient)
        Alpha = np.diag(Alpha) + eta * d_alpha
        b = b + eta * d_b
        S = np.diag(S) + eta * d_slack
        Ksi = np.diag(Ksi) + eta * d_ksi

        # Counting iterations
        iteration += 1

        # Recalculating gap
        gap= S @ Alpha

        # As iterations go, we must allow objective function to get closer to the barrier
        mu = mu * 0.92

        print(f'Iteration {iteration} -- Current gap is :{gap} --  Mu is: {mu}')
        # time.sleep(0.1)
        gaps.append(gap)

        # go back to original format to new iteration
        Alpha = np.diag(Alpha)
        S = np.diag(S)
        Ksi = np.diag(Ksi)

    # plt.plot(gaps)
    # plt.show()

    # Plotting the support vectors

    
    colors = np.ones(Q.shape[0])
    alphas_boolean_mask = np.where(np.diag(Alpha) > threshold)
    colors[alphas_boolean_mask] = 2

    print(colors)

    df_plot = pd.DataFrame({'x1':X_train.T[0],'x2':X_train.T[1],'label':colors})
    p = (
        ggplot(data = df_plot, mapping = aes(x = 'x1', y = 'x2', fill = 'factor(label)')) + 
        geom_point()
    )
    print(p)

    return np.diag(Alpha), b, np.diag(S), np.diag(Ksi)

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

def dataset_viz(X, y):

    dataframe = pd.DataFrame({'X1':X[:,0],'X2':X[:,1],'y':np.ravel(y)})

    plot = (
        ggplot(data = dataframe, mapping = aes(x = 'X1', y = 'X2')) + 
        geom_point(aes(fill = 'factor(y)'))
    )

    return plot

if __name__ == '__main__':
    SIZE = 20
    train, test = simple_dataset(SIZE)

    X_train = train[:,[0,1]]
    y_train = train[:,2].reshape(-1,1)

    X_test = test[:,[0,1]]
    y_test = test[:,2].reshape(-1,1)

    # print(dataset_viz(X_train, y_train))
    # print(dataset_viz(X_test, y_test))

    Alpha, b, S, Ksi = primal_dual_path(X_train, y_train)

    # print(f'\n\nAlpha: \n{Alpha}')
    # print(f'\n\nb: \n{b}')
    # print(f'\n\nS: \n{S}')
    # print(f'\n\nKsi: \n{Ksi}')

    # classification Task

    all_labels, y_hats = classification(X_train = X_train, 
                                        y_train = y_train, 
                                        alphas = Alpha, 
                                        b = b[0], 
                                        X_test = X_test, 
                                        y_test = y_test)

    print(all_labels)

    y_true = np.ravel(y_test)

    print(accuracy_score(y_true = y_true, y_pred = y_hats))
