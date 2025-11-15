import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from InitializeDataframe import LoadDataframe

def GetDataset():
    dataPath = r'data\temporal_features_tsfel.csv'
    labelPath = r'data\label.csv'
    data, label = LoadDataframe(dataPath, labelPath)

    featureNames = ['X1', 'X4', 'X13']
    
    selector = SelectKBest(f_classif, k=3)
    dataset = selector.fit_transform(data, np.ravel(label))

    return dataset, label, featureNames

def SaveCSV(df):
    df.to_csv(r'data\SelectedFeatures.csv',index=False)

def CreateDf():
    dataset, label, featureNames = GetDataset()
    
    df = pd.DataFrame(data=dataset, columns=featureNames)
    #df['target'] = label
    
    SaveCSV(df)
