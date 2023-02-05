from numpy import array as ar 
import numpy as np
import cvxopt

from sklearn.metrics import accuracy_score

# A: equality constraints
# P: Positive definitive matrix
# For svm primal duah path formulated problem, q is the vectors of ones (which then is multiplied by alpha, the kkt multipliers)
def cvxopt_solve_qp(P, q, G, h, A=None, b=None):
    # https://scaron.info/blog/quadratic-programming-in-python.html
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
    if A is not None:
        args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return ar(sol['x']).reshape((P.shape[1],)), sol

def _gen_sample(mean, sd, size, target:int):

    x1 = np.random.normal(loc = mean, scale = sd, size = size)
    x2 = np.random.normal(loc = mean, scale = sd, size = size)

    target_array = ar([target]*size)

    sample = ar([x1,x2,target_array])    
    
    return sample.T

def simple_dataset(size, mean_pos:float = 10, mean_neg = 1):

    # Two-dimensional dataset
    # Negative examples with mean 0 and sd = 1
    # Positive examples with mean 10 and sd = 1
    
    neg_train = _gen_sample(mean_neg,1,size,-1)
    pos_train = _gen_sample(mean_pos,1,size,+1)

    train = np.concatenate((neg_train, pos_train), axis = 0)

    neg_test = _gen_sample(mean_neg,1,size,-1)
    pos_test = _gen_sample(mean_pos,1,size,+1)

    test = np.concatenate((neg_test, pos_test), axis = 0)

    return train,test


SIZE = 4
train, test = simple_dataset(SIZE)

X_train = train[:,[0,1]]
y_train = train[:,2].reshape(-1,1)

X_test = test[:,[0,1]]
y_test = test[:,2].reshape(-1,1)

polynomial_order = 2

Q = (y_train @ y_train.T) * (1+(X_train @ X_train.T))**polynomial_order
# print(Q)

npoints = X_train.shape[0]

e = np.ones(npoints).reshape(-1,1)

C = 2
upper_limit_vector = np.ones(npoints)*C

G = np.identity(npoints)

# yt@alpha = b (for svm)
# b = 0
# A yt
A = np.diag((y_train.T)[0])
b_vector = np.zeros(npoints)

alphas_result, sol = cvxopt_solve_qp(P = Q, q = -e, G = G, h = upper_limit_vector, A = A, b = b_vector)

threshold = 1E-16
boolean_mask = (alphas_result > threshold)
support_alphas = alphas_result[boolean_mask]
Y_margin = y_train[boolean_mask]
first_term = support_alphas*Y_margin
second_term = 1 + X_train[boolean_mask,:] @ (X_train[boolean_mask,:]).T**polynomial_order
multi_first_second = (first_term.T @ second_term)
b = Y_margin - multi_first_second
b_bias_term = np.mean(b)

def classification(all_alphas, X_train, y_train, X_test, polynomial_order, b_bias_term):

    all_predictions = []

    for i in range(X_test.shape[0]):

        alpha_y = all_alphas @ y_train
        kernel_term = (1 + (X_test[i,:] @ X_train.T)) ** polynomial_order
        
        label = sum(alpha_y * kernel_term) + b_bias_term

        if label >= 0:
            label = 1
        else:
            label = -1

        all_predictions.append(label)

    return all_predictions

all_predictions = classification(all_alphas = alphas_result, 
                                    X_train = X_train, 
                                    y_train = y_train, 
                                    b_bias_term = b_bias_term,
                                    X_test = X_test,
                                    polynomial_order=polynomial_order)


# plot3d(X_test[:,0], X_test[:,1], sum_labels)

y_true = np.ravel(y_test)
print(accuracy_score(y_true = y_true, y_pred = all_predictions))

