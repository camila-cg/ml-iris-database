'''
    Usar o dataset Iris para executar os algoritmos K-NN, Árvores de Decisão, NaiveBayes, SVM e MLP para construir classificadores.
    Separe o conjunto de dados iris em 2/3 para treinamento e 1/3 para teste. Use a mesma partição para todos os algoritmos.
    Variar o valor de K para 1, 3, 5, 7 e 9.
    Variar os parâmetros de SVM e MLP usando a função Randomized Search do Scikit (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
    ou o GridSearchCV (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
    Usar as implementações do Scikit. Podem ser criados os scripts no colab do Google e enviar o link.
    Atenção para dar permissão para o meu acesso. Outra opção é fazer upload do arquivo .ipynb.
    Analise os valores das métricas de acurácia (accuracy), precisão (precision), recall e F1.
'''
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def calcularMetricas(k, y_test, y_pred):
    scoring = {
        'k': k,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_micro': precision_score(y_test, y_pred, average = 'micro', zero_division=1),
        'precision_macro': precision_score(y_test, y_pred, average = 'macro', zero_division=1),
        'precision_weighted': precision_score(y_test, y_pred, average = 'weighted', zero_division=1),
        'recall_micro': recall_score(y_test, y_pred, average = 'micro', zero_division=1),
        'recall_macro': recall_score(y_test, y_pred, average = 'macro', zero_division=1),
        'recall_weighted': recall_score(y_test, y_pred, average = 'weighted', zero_division=1),
        'f1_micro': f1_score(y_test, y_pred, average = 'micro', zero_division=1),
        'f1_macro': f1_score(y_test, y_pred, average = 'macro', zero_division=1),
        'f1_weighted': f1_score(y_test, y_pred, average = 'weighted', zero_division=1)
    }
    return scoring

def classificadorKNN(X_train, X_test, y_train, y_test):
    print("classificadorKNN")
    valores_k = [1,3,5,7,9]
    scores = {}
    scores_list = []
    resultados = list()
    for k in valores_k:
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
        #TODO
        resultados.append(calcularMetricas(k, y_test, y_pred))

        print(accuracy_score(y_test, y_pred))
        scores[k] = accuracy_score(y_test, y_pred)
        scores_list.append(accuracy_score(y_test, y_pred))
    
    print(resultados)
    plt.plot(valores_k, scores_list)
    plt.xlabel('Valor de k para o KNN')
    plt.ylabel('Acurácia')
    plt.show()


def gridSearch(X_train, X_test, y_train, y_test):
    modelo = SVC(random_state=10, tol=0.0001)
    parameters = {
        'kernel':('linear', 'rbf', 'sigmoid', 'poly'), 
        'C':[0.025, 0.001, 0.05, 0.01, 0.1, 1, 4, 10, 50, 100, 1000],
        'gamma': [0.001, 0.01, 0.1, 1, 10, 50],
        'coef0': [0.1, 1, 10, 100],
        'degree': [1,2,3]
    }
    '''GridSearchCV(estimator, param_grid, scoring=None, fit_params=None, n_jobs=1, iid=True, refit=True, cv=None, verbose=0, 
    pre_dispatch=‘2*n_jobs’, error_score=’raise’, return_train_score=’warn’)'''

    clf = GridSearchCV(modelo, parameters, cv=10)
    clf.fit(X_train, y_train)
    #print(sorted(clf.cv_results_.keys()))
    print(clf.best_estimator_)
    print(clf)




def classificadorAD():
    print("classificadorAD")


def classificadorNB(X_train, X_test, y_train, y_test):
    print("classificadorNB")
    n_class=GaussianNB()
    n_class.fit(X_train, y_train)
    y_pred_bayes=n_class.predict(X_test)

def classificadorSVM(X_train, X_test, y_train, y_test):
    print("classificadorSVM")
    modelo=SVC()
    modelo.fit(X_train, y_train)
    pred=modelo.predict(X_test)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

    #Variar os parâmetros de SVM e MLP usando a função Randomized Search do Scikit (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
    #ou o GridSearchCV (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).


def classificadorMLP():
    print("classificadorMLP")

def classificadores(dataset):
    X=dataset.drop('class', axis=1)
    y=dataset['class']
    #print(y.head())
    #print(X.head())

    # Holdout com 1/3 dos dados para teste (50 registros)
    # e 2/3 para treinamento (100 registros)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50)
    
    #classificadorKNN(X_train, X_test, y_train, y_test)
    #classificadorSVM(X_train, X_test, y_train, y_test)
    gridSearch(X_train, X_test, y_train, y_test)
    

