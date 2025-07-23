import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from python_speech_features import mfcc

def ApplyClassificationAlgorithms():
    def ApplyAlgorithm(x_train, x_test, y_train, algorithm):
        classifier = algorithm
        classifier.fit(x_train, np.ravel(y_train))
        y_pred = classifier.predict(x_test)
        
        return y_pred

    def CalculateAccuracy(y_pred, y_test):
        cont = 0
        
        for i in range(len(y_pred)):
            if(y_test[i] == y_pred[i]):
                cont = cont + 1
                
        acc = cont/len(y_pred)
        
        return acc

    KNeighbors = KNeighborsClassifier()
    SupportVectorMachine = svm.SVC()
    RandomForest = RandomForestClassifier()
    algorithms = [KNeighbors, SupportVectorMachine, RandomForest]

    x_train = pd.read_csv(r'DataTrain\data_train.csv', usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    y_train = pd.read_csv(r'DataTrain\data_train.csv', usecols=[14])
    x_test = pd.read_csv(r'DataTrain\data_test.csv', usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    y_test = pd.read_csv(r'DataTrain\data_test.csv', usecols=[14])
    
    for algo in algorithms:
        y_pred = ApplyAlgorithm(x_train, x_test, y_train, algo)
        
        accuracy = CalculateAccuracy(y_pred, y_test)
        
        print(f"A taxa de acertos do modelo {algo} é {accuracy} \n")
        
ApplyClassificationAlgorithms()
