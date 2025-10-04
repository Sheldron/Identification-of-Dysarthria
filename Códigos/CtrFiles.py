def ClearFile(path):
    f = open(path, "w")
    f.close
    
def OpenFile(path):        
    f = open(path, "a")
        
    return f

def CloseFile(file):
    file.close