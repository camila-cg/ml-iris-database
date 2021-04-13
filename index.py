import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

def loadDatasetByUrl():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    dataset = pd.read_csv(url, names = attributes)
    return dataset

def loadDatasetBySklearn():
    data = load_iris()
    # https://numpy.org/devdocs/reference/generated/numpy.c_.html
    dataset = pd.DataFrame(data= np.c_[data['data'], data['target']], columns= data['feature_names'] + ['class'])
    #print(list(data.target_names))
    dataset['class'] = dataset["class"].replace(0, 'setosa')
    dataset['class'] = dataset["class"].replace(1, 'versicolor')
    dataset['class'] = dataset["class"].replace(2, 'virginica')
    #print(dataset.head())
    #print(dataset.groupby('class').size())
    return dataset


def plotBoxplot(dataset):
    fig, axs = plt.subplots(2, 2)
    #fn = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    #cn = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    cn = ['setosa', 'versicolor', 'virginica']
    sns.boxplot(x = 'class', y = 'sepal length (cm)', data = dataset, order = cn, ax = axs[0,0])
    sns.boxplot(x = 'class', y = 'sepal width (cm)', data = dataset, order = cn, ax = axs[0,1])
    sns.boxplot(x = 'class', y = 'petal length (cm)', data = dataset, order = cn, ax = axs[1,0])
    sns.boxplot(x = 'class', y = 'petal width (cm)', data = dataset,  order = cn, ax = axs[1,1])
    fig.tight_layout(pad=1.0)
    plt.show()

def plotHistogram(dataset):
    n_bins = 10
    fig, axs = plt.subplots(2, 2)
    axs[0,0].hist(dataset['sepal length (cm)'], bins = n_bins)
    axs[0,0].set_title('Sepal Length')
    axs[0,1].hist(dataset['sepal width (cm)'], bins = n_bins)
    axs[0,1].set_title('Sepal Width')
    axs[1,0].hist(dataset['petal length (cm)'], bins = n_bins)
    axs[1,0].set_title('Petal Length')
    axs[1,1].hist(dataset['petal width (cm)'], bins = n_bins)
    axs[1,1].set_title('Petal Width')
    fig.tight_layout(pad=1.0)
    plt.show()

def plotScatterPlot(dataset):
    # plt.scatter(x, y)
    sns.pairplot(dataset, hue="class", height = 2, palette = 'colorblind')
    plt.show()


def main():
    # 1. Carregue o dataset iris como um dataframe
    dataset = loadDatasetBySklearn()

    # 2. Calcule, para cada atributo, as estatísticas média, mediana, desvio-padrão, Q1, Q3, obliquidade e curtose. Apontem nos resultados de vocês qual(is) biblioteca(s) vcs utilizaram.
    print(dataset.describe())

    # 3. Desenhe boxplots para cada variável.
    plotBoxplot(dataset)

    # 4. Plote histogramas dos atributos e interprete sua distribuição.
    plotHistogram(dataset)

    # 5. Plote scatterplots e interprete sua distribuição, considerando as classes.
    plotScatterPlot(dataset)


main()