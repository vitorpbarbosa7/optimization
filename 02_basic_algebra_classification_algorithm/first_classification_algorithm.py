from typing import List, Dict, Callable, Type, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from plotnine import *

import warnings
warnings.filterwarnings('ignore')

def classif_algo(X_train, X_test, y_train, y_test):

    predicted_y = []
    true_y = []
    x1s = []
    x2s = []

    for index_test, unseen in enumerate(X_test):

        m_positive = 0
        m_negative = 0

        # Para somar o produto interno de exemplos nao vistos pelo modelo 
        # contra todos os outros pontos contidos nos conjuntos de negativo e positivo
        sum_positive = 0
        sum_negative = 0

        # Aplicar as equacoes para exemplos de teste

        for index_train, seen in enumerate(X_train):

            # classe positiva
            if y_train[index_train] == +1:
                single_dot_product_positive = np.dot(unseen, seen)
                sum_positive += single_dot_product_positive
                m_positive += 1

            # classe negativa
            if y_train[index_train] == -1:
                single_dot_product_negative = np.dot(unseen, seen)
                sum_negative += single_dot_product_negative
                m_negative += 1

        # To compute the b term:

        m_squared_positive = 0
        m_squared_negative = 0

        # Para computar o produto interno de exemplos nao vistos contra exemplos contidos na classe positiva e negativa
        sum_b_positive = 0
        sum_b_negative = 0

        # Computar b, de acordo com os produtos internos
        for i, seen_i in enumerate(X_train):
            for j, seen_j in enumerate(X_train):

                if y_train[i] == -1 & y_train[j] == -1:
                    single_dot_product_negative_x_x = np.dot(seen_i, seen_j)
                    sum_b_negative += single_dot_product_negative_x_x
                    m_squared_negative += 1

                if y_train[i] == -1 & y_train[j] == +1:
                    single_dot_product_positive_x_x = np.dot(seen_i, seen_j)
                    sum_b_positive += single_dot_product_positive_x_x
                    m_squared_positive += 1

        # Finalmente o b sera:

        b = 1/2 * (
                    (
                        (1/m_squared_negative) * sum_b_negative
                    ) - \
                    (
                        (1/m_squared_positive) * sum_b_positive
                    )
        )


        # y pode finalmente ser computado:

        y = np.sign(
            (
                (1/m_positive) * sum_positive
            ) - \
            (
                (1/m_negative) * sum_negative
            ) + \
            (
                b
            )
        )

        predicted_y.append(y)
        true_y.append(y_test[index_test])
        x1s.append(unseen[0])
        x2s.append(unseen[1])

    results = pd.DataFrame({'predicted_y':predicted_y,'true_y':true_y,'X1':x1s,'X2':x2s})
    
    return results

def generate_2d_points(loc:int = 0, scale:int = 1, size:int = 100) -> List[Union[int, float]]:

    pair_points = [
        list(np.random.normal(loc = loc, scale = scale, size = size)),
        list(np.random.normal(loc = loc, scale = scale, size = size))
        ]

    return pair_points


def main():

    SIZE = 100

    # pos
    pos_dataset = generate_2d_points(0,1,SIZE)
    pos_dataframe = pd.DataFrame({'X1':pos_dataset[0],'X2':pos_dataset[1],'y':[+1]*SIZE})
    # neg
    neg_dataset = generate_2d_points(10,1,SIZE)
    neg_dataframe = pd.DataFrame({'X1':neg_dataset[0],'X2':neg_dataset[1],'y':[-1]*SIZE})

    dataframe = pos_dataframe.append(neg_dataframe)
    dataframe = dataframe.reset_index()

    X = dataframe[['X1','X2']]
    y = dataframe['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=7)

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    results = classif_algo(X_train, X_test, y_train, y_test)

    print(results)

    results['predicted_y'] = results['predicted_y'].astype('category')

    p = (
        ggplot(data = results, mapping = aes(x = 'X1', y = 'X2', fill = 'predicted_y')) 
        + geom_point()
    )
    print(p)
    # plt.scatter(x = pos_dataset_test[0],y = pos_dataset_test[1], marker = 'o', c = 'green')
    # plt.scatter(x = neg_dataset_test[0],y = neg_dataset_test[1], marker = 'x', c = 'red')
    # plt.show()
    
    return None


if __name__ == '__main__':
    main()