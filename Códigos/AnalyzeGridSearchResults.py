import numpy as np
import pandas as pd

def AnalyzeGridSearchResults():
    def ClearFile():
        f = open(r"ResultadosGridSearch\BestParamsPerModel.txt", "w")
        f.close
    
    def OpenFile():        
        f = open(r"ResultadosGridSearch\BestParamsPerModel.txt", "a")
        
        return f
        
    def OpenDataFrame():
        results = pd.read_csv(r'ResultadosGridSearch\ResultadosGrisSearch.csv')

        results = results.drop('index', axis=1)
        
        return results

    def SplitDfPerModel(df):
        resultsKNN = df.loc[df['MODEL'] == "KNN"]
        resultsSVM = df.loc[df['MODEL'] == "SVM"]
        resultsRDF = df.loc[df['MODEL'] == "RDF"]

        resultsModels = [resultsKNN, resultsSVM, resultsRDF]

        return resultsModels

    def GetBestParams(df, scores):
        f = OpenFile()
        f.write("Todas as informações de cada ieracao com os melhores parametros: \n")
        
        for m in df:
            for s in scores:
                f.write(str(m.loc[m[s] == m[s].min()]))
        
            f.write(f"\n===============================================\n")
                
        f.close()
        
    def GetInfoScores(df, models, scores):
        f = OpenFile()
        f.write("\nMedia e Desvio Padrao por metrica: \n\n")
        
        for m in df:
            for s in range(len(scores)):
                f.write(str(f"A media de {scores[s]} para o modelo {models[s]} eh {m[scores[s]].mean()} \n"))
                f.write(str(f"O desvio padrao de {scores[s]} para o modelo {models[s]} eh {m[scores[s]].std()} \n\n"))
            
            f.write(f"\n===============================================\n")
        
        f.close()
    
    ClearFile()
        
    results = OpenDataFrame()
    
    resultsModels = SplitDfPerModel(results)
    
    models = ["KNN", "SVM", "RDF"]
    scores = ["F1", "ACCURACY", "RECALL"]
    
    GetBestParams(resultsModels, scores)
    
    GetInfoScores(resultsModels, models, scores)

AnalyzeGridSearchResults()