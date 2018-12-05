import pandas as pd

#import ML support libraries
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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
    
    
#Test your code
nbaData = loadData("nba_2013_clean.csv")
knnModel = runKNN(nbaData, "pos", "player")
#^predict position, ignore player 
classifyPlayer(nbaData.loc[nbaData['player'] == 'LeBron James'], nbaData, knnModel, 'pos', 'player')
#locate the row where my player value equals Lebron and pass the rest of the arguments for the function 

################################################################################
# WEDNESDAY 12.5.18
################################################################################
#PROBLEM 2. Update your k-NN code such that your runKNN function takes an additional parameter standing for the number of neighbors to consider and uses this parameter to perform a classification. 

#Look above 

#2. Run the classification with a 60/40 split (60% training data, 40% testing data) using 5 neighbors. Compute the F1 score (hint: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) and the accuracy score for the data. What do these scores tell you about using k-NN to classify a player's position based on their statistics? 

#3. Create a new function runKNNCrossfold that takes two arguments, one being the dataset and a second being a k-value standing for the number of folds. Update your classifier code to use k-fold cross-validation. Note that you can set up this cross validation manually or experiment with SciKit Learn's built-in methods for doing this. Print the accuracy for each fold for k equal to 5, 7, and 10.

#4. Write a function called determineK which takes a dataset as an argument. Use this function to determine what the optimal setting of k is for kNN, where the best k is the one that maximizes the mean accuracy in your crossfold validation. Print this k and the resulting accuracy. 

################################################################################
# FRIDAY 12.7.18 
################################################################################