import numpy as np
import pandas as pd
import time
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score,recall_score,make_scorer

def ApplyGridSearch(classifiers):
    Xt = pd.read_csv(r'DataTrain\data_train.csv', usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    yt = pd.read_csv(r'DataTrain\data_train.csv', usecols=[14])
    
    list_models=[]

    #mínimo 30 iterações
    for run in range(2):
        for name, clf_name, clf, clf_param_grid in classifiers:

            #n_folds = 5

            #instancia método de validação cruzada
            #cv = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=run)
            cv = model_selection.LeaveOneOut()

            #instancia o Grid Search: classificador, grade de parâmetros, validacao cruzada, score
            clf_gcv=model_selection.GridSearchCV(estimator=clf, param_grid=clf_param_grid, cv=cv, scoring=make_scorer(accuracy_score), n_jobs=-1, verbose=0)

            #atribui o tempo quando inicia o processo de divisão em treinamento e teste
            st = time.time()

            #divide treinameto e teste
            for train, test in cv.split(Xt):
                X_train, X_test = Xt.values[train], Xt.values[test]
                y_train, y_test = yt.values[train], yt.values[test]

            #treina
            print("Começou o treino")
            clf_gcv.fit(X_train, np.ravel(y_train))

            #imprime melhores parâmetros
            #print(clf_gcv.best_params_)

            #realiza a predição
            y_pred = clf_gcv.predict(X_test)

            #calcula o tempo final
            st = time.time()-st

            #imprime metricas
            print(f"\nA iteração é {run}, o modelo é {clf_name}. \n \
                        O f1 score é {f1_score(y_test, y_pred, average='weighted')}, \
                        a acurácia é {accuracy_score(y_test, y_pred)}, \
                        o recall score é {recall_score(y_test, y_pred, average='weighted')}. \n \
                        O tempo decorrido foi {st} \n \
                        Os melhores parâmetros são {clf_gcv.best_params_}")
            print("\n====================================================\n")

            #armazena infromações em um dicionário
            l = {
                'DATASET'	: "Dysarthria"						,
                'MODEL'		: clf_name						,
                'RUN'		: run							,
                'BEST_PARAMS'	: clf_gcv.best_params_			 		,
                'TIME'		: st							,
                'Y_TRUE'     	: y_test						,
                'Y_PRED'     	: y_pred						,
                'F1'     	: f1_score(y_test, y_pred, average='weighted')		,
                'ACCURACY'     	: accuracy_score(y_test, y_pred)			,
                'RECALL'     	: recall_score(y_test, y_pred, average='weighted')	,
                }
            #adiciona a cada rodada em uma lista
            list_models.append(l)
    
    #salva os resultados em um dataframe
    return pd.DataFrame(list_models)

classifiers = [
	(
	"Nearest Neighbors",'KNN',		KNeighborsClassifier(),
	{'n_neighbors':[1,2,3,4,5,6,10], 'weights':['uniform', 'distance']}
    ),
    
    (
	"SVM",'SVM',			SVC(),
	{'kernel': ['linear','rbf'],'gamma':  [0.01,0.1,1],'C': [1,10,100]}#tipo de kernel e parametro de penalidade
    ),

    #(
    #'Random Forest','RDF',    RandomForestClassifier(),
	#{'hidden_layer_sizes':[[5],[5,5],[5,5,5],[5,5,5,5],[10],[10,10],[10,10,10],[50]],
	#'activation':['identity', 'logistic', 'tanh', 'relu']},
    #),
  ]

ResultsGridSearch = ApplyGridSearch(classifiers)

model = ['KNN', 'SVM'] #'RDF'
acc = []
results = []
bestParams = []

for i in range(len(model)):
    results.append(ResultsGridSearch[ResultsGridSearch['MODEL'] == model[i]])
    acc.append(results[i]['ACCURACY'].mean())
    bestParams.append(results[i][results[i].index == 2]['BEST_PARAM'].values)
    
    print(f"\nOs melhores parâmetros para o modelo {model[i]} são {bestParams[i]}\n")
    print(f"A acurácia do modelo é {acc[i]}\n")




"""ResultsKNN = ResultsGridSearch[ResultsGridSearch['MODEL'] == 'KNN']
ResultsSVM = ResultsGridSearch[ResultsGridSearch['MODEL'] == 'SVM']
#ResultsRDF = ResultsGridSearch[ResultsGridSearch['MODEL'] == 'RDF']

AccKNN = ResultsKNN['ACCURACY'].mean()
AccSVM = ResultsSVM['ACCURACY'].mean()
#AccRDF = ResultsRDF['ACCURACY'].mean()

BestParamsKNN = ResultsKNN[ResultsKNN.index == 2]["BEST_PARAMS"].values
BestParamsSVM = ResultsSVM[ResultsSVM.index == 2]["BEST_PARAMS"].values
#BestParamsRDF = ResultsRDF[ResultsRDF.index == 2]["BEST_PARAMS"].values"""