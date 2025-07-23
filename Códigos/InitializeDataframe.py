import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

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