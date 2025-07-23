import numpy as np
import pandas as pd
from scipy import stats
from sklearn import datasets

import time

from sklearn import preprocessing
from sklearn import decomposition
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score,make_scorer

def InitializeDataframe():
        def LoadDataframe():
            data = pd.read_csv(r'data\temporal_features_tsfel.csv')
            label = pd.read_csv(r'data\label.csv')
            
            return data, label

        def SplitDatabase(data, label, testSize):
            x_train, x_test, y_train, y_test, = train_test_split(data, label, test_size = testSize)
            
            return x_train, x_test, y_train, y_test

        data, label = LoadDataframe()

        x_train, x_test, y_train, y_test = SplitDatabase(data, label, testSize=0.20)
        y_test = np.ravel(y_test)
        
        return data, label, x_train, x_test, y_train, y_test

def SaveTrainTestToCSV(data, x_train, x_test, y_train, y_test):
    df_train=pd.DataFrame(x_train,columns=data.columns)
    df_train['Class'] = y_train
    df_train.to_csv(r'DataTrain\data_train.csv',index=False)
    
    df_test=pd.DataFrame(x_test,columns=data.columns)
    df_test['Class'] = y_test
    df_test.to_csv(r'DataTrain\data_test.csv',index=False)
    
data, label, x_train, x_test, y_train, y_test = InitializeDataframe()

SaveTrainTestToCSV(data, x_train, x_test, y_train, y_test)