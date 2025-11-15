import pandas as pd
from CtrFiles import ClearFile, OpenFile, CloseFile, BreakLineFile

def OpenDataFrame():
    results = pd.read_csv(r'Resultados\ResultadosGrisSearch.csv')

    results = results.drop('index', axis=1)
        
    return results

def SplitDfPerModel(df):
    resultsKNN = df.loc[df['MODEL'] == "KNN"]
    resultsSVM = df.loc[df['MODEL'] == "SVM"]
    resultsRDF = df.loc[df['MODEL'] == "RDF"]

    resultsModels = [resultsKNN, resultsSVM, resultsRDF]

    return resultsModels

def GetBestParams(df, scores, pathFile):
    f = OpenFile(pathFile)
    f.write("Todas as informações de cada iteracao com os melhores parametros: \n")
        
    for m in df:
        for s in scores:
            f.write(str(m.loc[m[s] == m[s].max()]))
        
        BreakLineFile(f)
                
    CloseFile(f)
        
def GetInfoScores(df, models, scores, pathFile):
    f = OpenFile(pathFile)
    f.write("\nMedia e Desvio Padrao por metrica: \n\n")
        
    for contador, m in enumerate(df):
        for s in range(len(scores)):
            f.write(str(f"A media de {scores[s]} para o modelo {models[contador]} eh {m[scores[s]].mean()} \n"))
            f.write(str(f"O desvio padrao de {scores[s]} para o modelo {models[contador]} eh {m[scores[s]].std()} \n\n"))
            
        BreakLineFile(f)
        
    CloseFile(f)
    
def AnalyzeGridSearchResults(): 
    pathFile = r"Resultados\BestParamsPerModel.txt"
    
    ClearFile(pathFile)
    
    results = OpenDataFrame()
    
    resultsModels = SplitDfPerModel(results)
    
    models = ["KNN", "SVM", "RDF"]
    scores = ["F1", "ACCURACY", "RECALL"]
    
    GetBestParams(resultsModels, scores, pathFile)
    
    GetInfoScores(resultsModels, models, scores, pathFile)

AnalyzeGridSearchResults()