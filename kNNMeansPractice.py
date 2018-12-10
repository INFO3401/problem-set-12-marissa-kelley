import pandas as pd
import seaborn as sns
#import ML support libraries
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

################################################################################
# MONDAY 12.3.18
################################################################################
#PROBLEM 1. Classify a player's position using k-NN and the provided classification data. 
def loadData(datafile):
    with open(datafile, 'r', encoding = "latin1") as csvfile: #r means read only 
        data = pd.read_csv(csvfile)
        
        #Inspect the data
        print(data.columns.values)
        
        return data

def runKNN(dataset, prediction, ignore):
    #ignore is the list of things you don't want to consider as part of your data
    #SET UP OUR DATASET
    X = dataset.drop(columns=[prediction,ignore])
    #prediction data will be considered in our training but will be useless if we get new data, 
    #so we ignore it
    Y = dataset[prediction].values
    #neighbors = dataset
    
    #Use train split thing that splits the data into a training and testing set
    #need to specify test size (what proportion of dataset do I actually want to consider) i.e 20% of data
    #random state-- am I selecting them randomly
    #Stratify try to get a nice uniform sampling of all the different possible settings 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1, stratify = Y)
    
    #RUn a k-NN algorithm
    #builds all classifiers as objects
    #n_neighbors is setting what we want as k
    knn = KNeighborsClassifier(n_neighbors=5)
    #after instantiating model, we want to be able to train the model
    #build the model using feature values to try and predict my dependent variable 
    knn.fit(X_train, Y_train)
    
    #test the model: 
    score = knn.score(X_test, Y_test)
    print("Predicts " + prediction + " with " + str(score) + " accuracy")
    print("Chance is: " + str(1.0/len(dataset.groupby(prediction))))
    
    return knn 

def classifyPlayer(targetRow, data, model, prediction, ignore):
    X = targetRow.drop(columns=[prediction, ignore])
    
    #Determine the five closest neighbors to our target row
    neighbors = model.kneighbors(X, n_neighbors=5, return_distance = False)
    
    #Print out the neighbors data
    for neighbor in neighbors[0]:
        print(data.iloc[neighbor]) #iloc is the index location 

def runKMeans(dataset, ignore):
    #Set up the dataset
    X = dataset.drop(columns = ignore)
    
    #Run K-means algorithm
    #choose k here, remember for kmeans is increase k until we stop getting noticeble gains 
    kmeans = KMeans(n_clusters = 5)
    
    #Train the model
    kmeans.fit(X)
    
    #Add the prediction directly to the dataframe
    #what cluster does this fall into? 0-4 from the 5 clusters
    #predict x variable, line up indicies with my indicies in my dataframe 
    dataset['cluster'] = pd.Series(kmeans.predict(X), index = dataset.index)
    
    #Scatterplot Matrix 
    #what features divide out well, which ones don't
    #map a different color to each of my clusters
    #palette says is go and use the color palette in seaborn called set 2
    scatterMatrix = sns.pairplot(dataset.drop(columns = ignore), hue = 'cluster', palette = 'Set2')
    
    scatterMatrix.savefig("kmeansClusters.png")
    #will save it as a static visualization 
    #could build this out into Altair
    
    return kmeans
    
#Test your code
nbaData = loadData("nba_2013_clean.csv")
knnModel = runKNN(nbaData, "pos", "player")
#^predict position, ignore player 
classifyPlayer(nbaData.loc[nbaData['player'] == 'LeBron James'], nbaData, knnModel, 'pos', 'player')
#locate the row where my player value equals Lebron and pass the rest of the arguments for the function

kmeansModel = runKMeans(nbaData, ['pos', 'player']) 