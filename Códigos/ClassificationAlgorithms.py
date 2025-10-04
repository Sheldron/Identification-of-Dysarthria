import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score
from CtrFiles import ClearFile, OpenFile, CloseFile

def SetFile():
    pathFile = r"Resultados\ClassificacaoResultados.txt"
    ClearFile(pathFile)
    
    return pathFile

def ConfigModels():
    KNeighbors = KNeighborsClassifier(n_neighbors=10, weights="distance")
    SupportVectorMachine = svm.SVC(C=10, gamma=0.01, kernel="rbf")
    RandomForest = RandomForestClassifier(max_depth=None, min_samples_split=2, n_estimators=200)
    algorithms = [KNeighbors, SupportVectorMachine, RandomForest]
        
    return algorithms
    
def GetDataFromDf():
    x_train = pd.read_csv(r'DataTrain\data_train.csv', usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    y_train = pd.read_csv(r'DataTrain\data_train.csv', usecols=[14])
    x_test = pd.read_csv(r'DataTrain\data_test.csv', usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    y_test = pd.read_csv(r'DataTrain\data_test.csv', usecols=[14])
        
    return x_train, y_train, x_test, y_test
    
def TrainAlgorithm(x_train, x_test, y_train, algorithm):
    classifier = algorithm
    classifier.fit(x_train, np.ravel(y_train))
    y_pred = classifier.predict(x_test)
        
    return y_pred

def CalculateScores(y_test, y_pred):
    f1 = f1_score(y_test, y_pred, average="weighted")
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="weighted")
        
    return [f1, accuracy, recall]

def SaveResults(pathFile, algo, scores, contador):
    f = OpenFile(pathFile)
            
    f.write(f"Para o modelo {algo}, \nO F1 score eh {scores[contador][0]} \nA accuracy eh {scores[contador][1]} \nO recall score eh {scores[contador][2]}\n")
    f.write("\n=================================================\n")
            
    CloseFile(f)

def ApplyAlgorithms():
    scores = []
    contador = 0
    
    pathFile = SetFile()
    algorithms = ConfigModels()
    x_train, y_train, x_test, y_test = GetDataFromDf()
        
    for algo in algorithms:
        print(f"Iniciando treino do modelo {algo}!")
        y_pred = TrainAlgorithm(x_train, x_test, y_train, algo)
            
        scores.append(CalculateScores(y_test, y_pred))
        
        SaveResults(pathFile, algo, scores, contador)
        contador += 1
        
        print(f"Modelo {algo} treinado! \n")
        print("\n============================\n")

ApplyAlgorithms()            

