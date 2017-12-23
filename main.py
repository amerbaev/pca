from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits, load_breast_cancer, load_wine
import numpy as np


def kaiser(vals):
    vals = list(filter(lambda x: x > 1, vals))
    return len(vals)


def broken_stick(vals):
    n = len(vals)
    trC = sum(vals)
    n_comp = 0
    for i in range(n):
        l = sum([1/j for j in range(i+1, n+1)])/n
        if vals[i] / trC > l:
            n_comp += 1
        else:
            break
    return n_comp


if __name__ == '__main__':
    X = load_digits().data
    print('Размерность данных:', len(X[0]))
    X_cent = StandardScaler().fit_transform(X)
    pca = PCA(svd_solver='full').fit(X_cent)
    print('Главных компонент по правилу кайзера: ', kaiser(pca.explained_variance_))
    print('Главных компонент по правилу сломанной трости: ', broken_stick(pca.explained_variance_))

