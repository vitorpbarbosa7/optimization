# https://www.amazon.com.br/Machine-Learning-Practical-Approach-Statistical/dp/3319949888

from typing import List, Dict, Callable, Type, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from plotnine import *

import warnings
warnings.filterwarnings('ignore')

# import logging
# log = logging.getLogger(__name__)
# handler = logging.StreamHandler()
# handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
# log.setLevel(logging.INFO)
# print('log is working')   

def classif_algo(X_train, X_test, y_train, y_test):

    print('a')

    predicted_y = []
    true_y = []
    X1s = []
    X2s = []

    for index_test, unseen in enumerate(X_test):
        print('b')

        m_positive = 0
        m_negative = 0

        # Para somar o produto interno de exemplos nao vistos pelo modelo 
        # contra todos os outros pontos contidos nos conjuntos de negativo e positivo
        sum_positive = 0
        sum_negative = 0

        # Aplicar as equacoes para exemplos de teste

        for index_train, seen in enumerate(X_train):
            print('c')

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
            print('d')
            for j, seen_j in enumerate(X_train):
                print('e')

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
        X1s.append(unseen[0])
        X2s.append(unseen[1])

    print('z')
    results = pd.DataFrame({'predicted_y':predicted_y,'true_y':true_y,'X1':X1s,'X2':X2s})
    
    return results

def generate_2d_points(mean:int = 0, sd:int = 1, size:int = 100) -> List[Union[int, float]]:

    pair_points = [
        list(np.random.normal(loc = mean, scale = sd, size = size)),
        list(np.random.normal(loc = mean, scale = sd, size = size))
        ]

    return pair_points

def make_points(size:int = 200):

    X_small, y_small = make_circles(n_samples=(250,500), random_state=2, 
    noise=0.04, factor = 0.3)
    X_large, y_large = make_circles(n_samples=(250,500), random_state=2, 
    noise=0.04, factor = 0.7)

    y_large[y_large==1] = 2
    df = pd.DataFrame(np.vstack([X_small,X_large]),columns=['X1','X2'])
    df['y'] = np.hstack([y_small,y_large])
    print(df.y.value_counts())

    df = df[df['y'].isin([0,1])]

    df['y'] = np.where(df['y']==0, -1, df['y'])

    pos_dataset = df[df['y']==+1].values
    neg_dataset = df[df['y']==-1].values

    print(len(pos_dataset))
    print(len(neg_dataset))

    plot = (
        ggplot(data = df, mapping = aes(x = 'X1', y = 'X2', fill = 'factor(y)')) + 
        geom_point()
    )
    # print(plot)

    return df

def main():

    SIZE = 200

    dataframe = make_points(SIZE)

    X = dataframe[['X1','X2']]
    y = dataframe['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=7)
    print(f'Splitted')

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    print(f'Entering classif algorithm...')
    results = classif_algo(X_train, X_test, y_train, y_test)

    print(results)

    results['predicted_y'] = results['predicted_y'].astype('category')

    p = (
        ggplot(data = results, mapping = aes(x = 'X1', y = 'X2', fill = 'predicted_y')) 
        + geom_point()
    )
    print(p)
    
    return None


if __name__ == '__main__':
    main()
