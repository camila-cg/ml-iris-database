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
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def calcularMetricas(y_test, y_pred):
  scoring = {
    #'k': k,
    'accuracy': [accuracy_score(y_test, y_pred)],
    'precision_micro': [precision_score(y_test, y_pred, average = 'micro', zero_division=1)],
    'precision_macro': [precision_score(y_test, y_pred, average = 'macro', zero_division=1)],
    'precision_weighted': [precision_score(y_test, y_pred, average = 'weighted', zero_division=1)],
    'recall_micro': [recall_score(y_test, y_pred, average = 'micro', zero_division=1)],
    'recall_macro': [recall_score(y_test, y_pred, average = 'macro', zero_division=1)],
    'recall_weighted': [recall_score(y_test, y_pred, average = 'weighted', zero_division=1)],
    'f1_micro': [f1_score(y_test, y_pred, average = 'micro', zero_division=1)],
    'f1_macro': [f1_score(y_test, y_pred, average = 'macro', zero_division=1)],
    'f1_weighted': [f1_score(y_test, y_pred, average = 'weighted', zero_division=1)]
  }
  return scoring


def classificadorKNN(X_train, X_test, y_train, y_test):
  print("classificadorKNN")
  valores_k = [1,3,5,7,9]
  scores = list()
  for k in valores_k:
    modelo = KNeighborsClassifier(n_neighbors = k)
    #print(modelo.get_params().keys())
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    metricas = calcularMetricas(k, y_test, y_pred)
    scores.append(metricas)
    print(metricas)
    print(metricas['accuracy'])
   
  resultados = pd.DataFrame (data = scores)
  resultados.to_csv('./result_knn.csv', encoding='utf-8')
  #plt.plot(valores_k, scores)
  #plt.xlabel('Valor de k para o KNN')
  #plt.ylabel('Acurácia')
  #plt.show()


def classificadorSVM(X_train, X_test, y_train, y_test):
  print("SVM:")
  modelo = SVC(random_state=10, tol=0.0001)
  #print(modelo.get_params().keys())

  param_grid = [
    {'C':[0.025, 0.001, 0.05, 0.01, 0.1, 1, 4, 10, 50, 100, 1000], 'kernel': ['linear']},
    {'C':[0.025, 0.001, 0.05, 0.01, 0.1, 1, 4, 10, 50, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1, 10, 50], 'kernel': ['rbf']},
    {'C':[0.025, 0.001, 0.05, 0.01, 0.1, 1, 4, 10, 50, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1, 10, 50], 'coef0': [0.1, 1, 10, 100], 'degree': [1,2,3], 'kernel': ['poly']},
    {'C':[0.025, 0.001, 0.05, 0.01, 0.1, 1, 4, 10, 50, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1, 10, 50], 'coef0': [0.1, 1, 10, 100], 'kernel': ['sigmoid']}
  ]

  scoring = {
    'accuracy': 'accuracy',
    'precision_micro': make_scorer(precision_score, average = 'micro', zero_division=1),
    'precision_macro': make_scorer(precision_score, average = 'macro', zero_division=1),
    'recall_micro': 'recall_micro',
    'recall_macro': 'recall_macro',
    'f1_micro': 'f1_micro',
    'f1_macro': 'f1_macro'
  }

  '''
  Fitting 10 folds for each of 3168 candidates, totalling 31680 fits
  Melhor modelo:  SVC(C=0.025, coef0=0.1, degree=2, gamma=50, kernel='poly', random_state=10, tol=0.0001)
  Melhor acurácia:  0.97

  Fitting 10 folds for each of 1133 candidates, totalling 11330 fits
  Melhor modelo:  SVC(C=0.05, coef0=100, gamma=0.001, kernel='poly', random_state=10, tol=0.0001)
  Melhor acurácia:  0.99
  '''

  '''
  GridSearchCV(estimator, param_grid, scoring=None, fit_params=None, n_jobs=1, iid=True, refit=True, cv=None, verbose=0, 
  pre_dispatch=‘2*n_jobs’, error_score=’raise’, return_train_score=’warn’)
  '''

  clf = GridSearchCV(modelo, param_grid, scoring=scoring, refit='accuracy', cv=10, verbose=1)
  clf.fit(X_train, y_train)
  #print(sorted(clf.cv_results_.keys()))
  print('Melhor modelo: ', clf.best_estimator_)
  print("Melhor acurácia: ", clf.best_score_)
    
  pd.set_option('max_columns', 200)
  result_grid_search = pd.DataFrame(clf.cv_results_)
  pd.DataFrame(data = result_grid_search).to_csv('result_grid_search_svm.csv', encoding='utf-8')

  #modelo.fit(X_train, y_train)
  #pred=modelo.predict(X_test)
  #print(confusion_matrix(y_test, pred))
  #print(classification_report(y_test, pred))


def classificadorAD(X_train, X_test, y_train, y_test):
  print("Árvores de Decisão:")
  modelo = DecisionTreeClassifier(max_depth = 3, random_state = 42)
  modelo.fit(X_train, y_train)

  feature_names = X_train.columns
  labels = y_train.unique()

  # Plotando a árvore de decisão
  # TODO: SALVAR
  plt.figure(figsize=(30,10))
  a = plot_tree(modelo, feature_names = feature_names, class_names = labels, rounded = True, filled = True, fontsize=14)
  #plt.show()

  # Exibindo diagrama da árvore em texto
  tree_rules = export_text(modelo, feature_names = list(feature_names))
  #print(tree_rules)

  y_pred = modelo.predict(X_test)
  matrix = confusion_matrix(y_test, y_pred)
  matrix_df = pd.DataFrame(matrix)

  # Calculando métricas
  metricas = calcularMetricas(y_test, y_pred)
  print(metricas)
  l = list()
  a.append(metricas)
  resultados = pd.DataFrame.from_dict(metricas)
  print(resultados)
  #resultados.to_csv('./result_ad.csv', encoding='utf-8')

  # Plotando matriz de confusão
  # TODO: SALVAR
  ax = plt.axes()
  sns.set(font_scale=1.3)
  plt.figure(figsize=(10,7))
  sns.heatmap(matrix_df, annot=True, fmt="g", ax=ax, cmap="magma")
  ax.set_title('Árvore de decisão - Matriz de confusão')
  ax.set_xlabel("Valor predito", fontsize =15)
  ax.set_xticklabels(['']+labels)
  ax.set_ylabel("Valor real", fontsize=15)
  ax.set_yticklabels(list(labels), rotation = 0)
  #plt.show()



def classificadorNB(X_train, X_test, y_train, y_test):
  print("classificadorNB")
  modelo=GaussianNB()
  modelo.fit(X_train, y_train)
  y_pred = modelo.predict(X_test) 
  print ("Accuracy : ", accuracy_score(y_test, y_pred))
  #comparativo = pd.DataFrame({'Valor Real':y_test, 'Valor Predito':y_pred})
  #print(comparativo)
  metricas = calcularMetricas(y_test, y_pred)
  print(metricas)
  pd.DataFrame(data = metricas).to_csv('result_nb.csv', encoding='utf-8')


def classificadorMLP():
  print("classificadorMLP")


def classificadores(dataset):
  X=dataset.drop('class', axis=1)
  y=dataset['class']

  # Holdout com 1/3 dos dados para teste (50 registros)
  # e 2/3 para treinamento (100 registros)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50)
  
  #classificadorSVM(X_train, X_test, y_train, y_test)
  #classificadorKNN(X_train, X_test, y_train, y_test)
  #classificadorAD(X_train, X_test, y_train, y_test)
  classificadorNB(X_train, X_test, y_train, y_test)
  #classificadorMLP(X_train, X_test, y_train, y_test)
  
    

