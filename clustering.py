import pandas as pd
import seaborn as sns
import numpy as np

#import ML support libraries
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

################################################################################
# MONDAY 12.3.18, WEDNESDAY 12.5.18, FRIDAY 12.7.18 
################################################################################

#Problems 1-3 and 6 were worked on with Taylor and Hannah. Problems 4, 5 and 7 were worked on with Jacob who walked us through his code. The text is from the three of us. 

#PROBLEM 1. Classify a player's position using k-NN and the provided classification data.

def loadData(datafile):
    with open(datafile, 'r', encoding = "latin1") as csvfile: #r means read only 
        data = pd.read_csv(csvfile)
        
        #Inspect the data
        print(data.columns.values)
        return data

#PROBLEM 2
def runKNN(dataset, prediction, ignore, neighbors):
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

#PROBLEM 3 (Pt. 1): test_size was originally 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state = 1, stratify = Y)
     
    
    #RUn a k-NN algorithm
    #builds all classifiers as objects
    #n_neighbors is setting what we want as k
    knn = KNeighborsClassifier(n_neighbors= neighbors)
    #after instantiating model, we want to be able to train the model
    #build the model using feature values to try and predict my dependent variable 
    knn.fit(X_train, Y_train)
    
    #Test the model: 
    score = knn.score(X_test, Y_test)
    Y_pred = knn.predict(X_test)
    print("Predicts " + prediction + " with " + str(score) + " accuracy")
    print("Chance is: " + str(1.0/len(dataset.groupby(prediction))))
    print ("F1 Score is: " + str(f1_score(Y_test, Y_pred, average = 'macro')))
    
    return knn 
#PROBLEM 3 (Pt. 2): 
#The F1 score and accuracy score were both low (roughly 45%) meaning using k-NN to classify a player's position based on their statistics only had a ~45% chance of being correct, and thus effective. For a F1 score to be effective, it would have to be closer to 1.  


def classifyPlayer(targetRow, data, model, prediction, ignore):
    X = targetRow.drop(columns=[prediction, ignore])
    
    Determine the five closest neighbors to our target row
    neighbors = model.kneighbors(X, n_neighbors=5, return_distance = False)
    
    #Print out the neighbors data
    for neighbor in neighbors[0]:
        print(data.iloc[neighbors]) #iloc is the index location 

#PROBLEM 4
def kNNCrossfold(dataset, prediction, ignore, neighbors):
    fold = 0 #makes a counter 
    accuracies = [] #makes an empty list
    kf = KFold(n_splits=neighbors)  #uses the kfold model from sklearn 

    #from previous problems: Setting up X and Y
    X = dataset.drop(columns=[prediction, ignore])
    Y = dataset[prediction].values

    for train,test in kf.split(X): #make a for loop for each kfold plit 
        fold += 1 #adds to the counter we've already made
        knn = KNeighborsClassifier(n_neighbors= neighbors) #uses the k_neighbors classifier for each input 
        knn = KNeighborsClassifier(n_neighbors = neighbors) #uses the kneighbors classifier for each of the k's (the 5,7,10).  
        knn.fit(X[train[0]:train[-1]], Y[train[0]:train[-1]]) #fit the data in order to train the classifier on each of the folds. It will remove the last one for testing. 

        pred = knn.predict(X[test[0]:test[-1]]) #knn.predict will predict the class labels for the provided the data. THis will predict the class labels for all the data in X from "column" 0 all the way to "column" -1, aka all but the last one because you're testing on the last one. 
        accuracy = accuracy_score(pred, Y[test[0]:test[-1]])
        accuracies.append(accuracy)
        print("Fold " + str(fold) + ":" + str(accuracy))

    return np.mean(accuracies)

        
#PROBLEM 5
def determineK(dataset, prediction, ignore, k_vals):
    best_k = 0 #used to create an integer value that gets replaced in the for loop
    best_accuracy = 0 #creates an integer value that gets replaced in the for loop
    
    for k in k_vals: #for loop in order to loop through the k values, 5,7 and 10
        current_k = kNNCrossfold(dataset, prediction, ignore, k) #runs the kNN cross validation function (from problem 4) for each k value in k_vals
        if current_k > best_accuracy: #if statement saying if the current k value was better than the best accuracy that's being stored. K is computed using the kNNCrossfold 
            best_k = k # if the k is better than the stored accuracy, then reset the variable k to the k value being looped through 
            best_accuracy = current_k #stores the current k value as the best accuracy 
    
    print("Best k, accuracy = " + str(best_k) + ", " + str(best_accuracy)) #prints the best k and its accuracy  
        

#PROBLEM 6        
def runKMeans(dataset, ignore, neighbors):
    #Set up the dataset
    X = dataset.drop(columns = ignore)
    
    #Run K-means algorithm
    #choose k here, remember for kmeans is increase k until we stop getting noticeble gains 
    kmeans = KMeans(n_clusters = neighbors)
    
    #Train the model
    kmeans.fit(X)
    
    #Add the prediction directly to the dataframe
    #what cluster does this fall into? 0-4 from the 5 clusters
    #predict x variable, line up indicies with my indicies in my dataframe 
    dataset['cluster'] = pd.Series(kmeans.predict(X), index = dataset.index)
    
    return kmeans

#PROBLEM 7
#From https://datascience.stackexchange.com/a/41125

def findClusterK(dataset, ignore):
    mean_distances = {} #create an empty dictionary
    X = dataset.drop(columns =ignore) #set up the dataset
    
    for n in np.arange(4,12):
        model = runKMeans(dataset, ignore, n) #run the model from problem 6
        mean_distances[n] = np.mean([np.min(x) for x in model.transform(X)]) 
        #use the .transform in order to get the distances of the points from all clusters. Use the list comprehension in order to get the minimum of those distances for each point in order to get the distance from the cluster that the point belongs to. Then take the mean of the list to get the average. 
    
    print("Best k by average distance: " + str(min(mean_distances, key=mean_distances.get))) #print the best k based on the average distance to the other points. Use the.get to return the value in the mean_distances key
    
    
#Scatterplot Matrix 
    #what features divide out well, which ones don't
    #map a different color to each of my clusters
    #palette says is go and use the color palette in seaborn called set 2
    #scatterMatrix = sns.pairplot(dataset.drop(columns = ignore), hue = #'cluster', palette = 'Set2')
    
    #scatterMatrix.savefig("kmeansClusters.png")
    #will save it as a static visualization 
    #could build this out into Altair
    
    #return kmeans

    
######Test your code######
nbaData = loadData("nba_2013_clean.csv")
knnModel = runKNN(nbaData, "pos", "player", 7)
#^predict position, ignore player 
#classifyPlayer(nbaData.loc[nbaData['player'] == 'LeBron James'], nbaData, knnModel, 'pos', 'player')
#locate the row where my player value equals Lebron and pass the rest of the arguments for the function

for k in [5,7,10]:
    print("Folds: " + str(k))
    kNNCrossfold(nbaData, "pos", "player", k)

determineK(nbaData, "pos", "player", [5,7,10])

kmeansModel = runKMeans(nbaData, ['pos', 'player'], 5)

findClusterK(nbaData, ['pos', 'player']) 