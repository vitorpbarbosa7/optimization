import numpy as np


def find_w_b(alphas, x, y):
    '''
    alphas: Initial kkt multipliers
    x: Input space of Variables
    y: Label of classification for each point of Rn
    '''
    # Vector w:
    # From derivative of Lagrangian /\(w,b,alpha) respect to w
    w = [0,0]
    for i in range(len(alphas)):
        w = w + alphas[i] * y[i] * x[i]
    
    # print(f'Coordenates for vector w: {w}')

    # Computing the bias term b

    negative_support_vector = x[y == -1]
    positive_support_vector = x[y == +1]
    difference_vector_pos_neg = (positive_support_vector - negative_support_vector).flatten()
    b = (-1/2)*(np.dot(w,difference_vector_pos_neg))

    # print(f'b bias term value: {b}')

    result = {}

    result['w'] = w
    result['b'] = b

    return result

def classify(w,b,unseen_x):

    classification = np.dot(w,unseen_x) + b

    return classification


if __name__ == '__main__':

    x = np.array([[+1,+1],[-1,-1]])
    y = np.array([+1,-1])

    # kkt multipliers
    # so, this was a initial guess, but how do I find those?
    alphas = np.array([0.05, 0.05])

    # testing some alpha values
    for i in range(10):
        alphas = alphas + 0.05 # makes objective function smaller 
        w_b_solution = find_w_b(alphas, x, y)

        w = w_b_solution['w']
        b = w_b_solution['b']

        # Objective function
        # Original primal problem original is of max margin: (2/(||w||**2)) -> becomes: min (1/2*(||w||**2))
        # Dual problem, must be maximized
        objective_W_alpha = 0
        
        for i in range(len(alphas)):
            for j in range(len(alphas)):
                objective_W_alpha += (-1/2) * \
                    (
                        alphas[i]*alphas[j]*y[i]*y[j]*np.dot(x[i],x[j])
                    )
        objective_W_alpha += sum(alphas)

        print(f'''
        This objective_W_alpha, which was calculated in this case, with alphas {alphas}, resulted in {objective_W_alpha} value. This is the term we want to maximize!!!
        ''')

        unseen_examples = [[+2,+2],[+3,+3],[+1.5,+1.5],[-3,-3],[-2,-2],[-1.5,-1.5]]
        for unseen_x in unseen_examples:
            result = classify(w,b,unseen_x)
            print(f'{unseen_x} is classified as {result}')        