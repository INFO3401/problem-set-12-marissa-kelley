import pandas as pd

#import ML support libraries
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def loadData(datafile):
    with open(datafile, 'r', encoding = "latin1") as csvfile: #r means read only 
        data = pd.read_csv(csvfile)
        
        #Inspect the data
        print(data.columns.values)
        
        return data

def runKNN(dataset, prediction, ignore):
    #ignore is the list of things you don't want to consider as part of your data
    
#Test your code
loadData("nba_2013_clean.csv")


