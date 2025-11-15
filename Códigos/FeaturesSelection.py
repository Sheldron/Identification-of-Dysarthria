import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe
from sklearn.feature_selection import f_classif
from InitializeDataframe import LoadDataframe
from FormatPrintText import BreakLine

def PrepareDf():
    dataPath = r'data\temporal_features_tsfel.csv'
    labelPath = r'data\label.csv'
    data, label = LoadDataframe(dataPath, labelPath)
    
    return data, label

def ConfigSelectors(): 
    kBest = SelectKBest(f_classif, k=3)
    percentile = SelectPercentile(f_classif, percentile=20)
    fpr = SelectFpr(f_classif, alpha=0.05)
    #fdr = SelectFdr(f_classif, alpha=0.01)
    #fwe = SelectFwe(f_classif, alpha=0.01)
    selectors = [kBest, percentile, fpr]
    
    return selectors

def ApplySelector(selector):
    filter = selector.get_support()
    featureNames = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14']
    selectedFeatues = np.array(featureNames)[filter]
    
    return selectedFeatues.tolist()

data, label = PrepareDf()
selectors = ConfigSelectors()
selectedFeatues = []

for s in selectors:
    BreakLine()
    print(f"Iniciando seleção do método {s} \n")
    data_new = s.fit_transform(data, np.ravel(label))
    selectedFeatues.append(ApplySelector(s))
    
    print(selectedFeatues)

