from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits, load_breast_cancer, load_wine
import matplotlib.pyplot as plt
import seaborn as sns


def kaiser(vals):
    mean = sum(vals) / len(vals)
    vals = list(filter(lambda x: x > mean, vals))
    return len(vals)


def broken_stick(vals):
    n = len(vals)
    trC = sum(vals)
    n_comp = 0
    for i in range(n):
        length = sum([1/j for j in range(i+1, n+1)])/n
        if vals[i] / trC > length:
            n_comp += 1
        else:
            break
    return n_comp


def process_dataset(x, name):
    print(name)
    print('Размерность данных:', len(x.data[0]))
    x_cent = StandardScaler().fit_transform(x.data)
    pca = PCA(svd_solver='full')
    pca.fit(x_cent)
    kaiser_size = kaiser(pca.explained_variance_)
    broken_size = broken_stick(pca.explained_variance_)
    print('Главных компонент по правилу кайзера: ', kaiser_size)
    print('Главных компонент по правилу сломанной трости: ', broken_size, end='\n\n')

    plt.legend(name)
    plt.xlabel('Номер компоненты')
    plt.ylabel('% дисперсии')
    plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_ * 100)
    plt.scatter(kaiser_size, pca.explained_variance_ratio_[kaiser_size] * 100)
    plt.scatter(broken_size, pca.explained_variance_ratio_[broken_size] * 100)
    sns.set()
    plt.show()


if __name__ == '__main__':
    for X, dset_name in ((load_digits(), 'Цифры'), (load_breast_cancer(), 'Рак'), (load_wine(), 'Вина')):
        process_dataset(X, dset_name)

