'''
    Usar o dataset Iris para executar os algoritmos K-NN, Árvores de Decisão, NaiveBayes, SVM e MLP para construir classificadores. 
    Separe o conjunto de dados iris em 2/3 para treinamento e 1/3 para teste.
    Use a mesma partição para todos os algoritmos. Variar o valor de K para 1, 3, 5, 7 e 9.
    Variar os parâmetros de SVM e MLP usando a função Randomized Search do Scikit (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
    ou o GridSearchCV (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
    Usar as implementações do Scikit. Podem ser criados os scripts no colab do Google e enviar o link.
    Atenção para dar permissão para o meu acesso. Outra opção é fazer upload do arquivo .ipynb.
    Analise os valores das métricas de acurácia (accuracy), precisão (precision), recall e F1.
'''
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def classificadorKNN():
    #TODO
    print("classificadorKNN")


def classificadorAD():
    #TODO
    print("classificadorAD")


def classificadorNB():
    #TODO
    print("classificadorNB")


def classificadorSVM(x_train, x_test, y_train, y_test):
    #TODO
    print("classificadorSVM")

def classificadorMLP():
    #TODO
    print("classificadorMLP")

def classificadores(dataset):
    X=dataset.drop('class', axis=1)
    y=dataset['class']
    print(y.head())
    print(X.head())

    #Holdout com 30% dos dados para teste
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.30)
    classificadorSVM(X_train, X_test, y_train, y_test)

