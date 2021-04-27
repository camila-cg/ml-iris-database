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
from sklearn.metrics import classification_report, confusion_matrix

def classificadorKNN():
    #TODO
    print("classificadorKNN")


def classificadorAD():
    #TODO
    print("classificadorAD")


def classificadorNB():
    #TODO
    print("classificadorNB")

#TODO
def classificadorSVM(X_train, X_test, y_train, y_test):
    print("classificadorSVM")
    #model=SVC()
    model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

    model.fit(X_train, y_train)
    pred=model.predict(X_test)
    print(confusion_matrix(y_test,pred))
    print(classification_report(y_test, pred))


def classificadorMLP():
    #TODO
    print("classificadorMLP")

def classificadores(dataset):
    X=dataset.drop('class', axis=1)
    y=dataset['class']
    #print(y.head())
    #print(X.head())

    #Holdout com 1/3 dos dados para teste (50 registros)
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=50)
    classificadorSVM(X_train, X_test, y_train, y_test)

