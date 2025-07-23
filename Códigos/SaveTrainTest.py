import pandas as pd

def SaveTrainTestToCSV(data, x_train, x_test, y_train, y_test):
    df_train=pd.DataFrame(x_train,columns=data.columns)
    df_train['Class'] = y_train
    df_train.to_csv(r'DataTrain\data_train.csv',index=False)
    
    df_test=pd.DataFrame(x_test,columns=data.columns)
    df_test['Class'] = y_test
    df_test.to_csv(r'DataTrain\data_test.csv',index=False)