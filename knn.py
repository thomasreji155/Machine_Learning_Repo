



# knn algorithm 
import numpy as np 
from collections import Counter 
class KNN:
    def __init__(self,k=0):
        self.k = k 
    def fit(self, X,y):
        self.X = X 
        self.y = y  


    def predict(self, X):      
        predicted_labels = [self._predict(x) for x in X] 
        return np.array(predicted_labels) 



    def euclidean_dist(self,vec1, vec2):
        return np.sqrt(np.sum((vec1 -vec2)**2)) 

    def _predict(self,x):
        # compute the distance 
        distance = [self.euclidean_dist(x,row) for row in self.X] 

        k_indexes = np.argsort(distance)[:self.k] 

        # get k nearest samples  get the labels also 
        k_nearest_labels = [self.y[i] for i in k_indexes]  


        # most common class label 
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
 
     

from sklearn.neighbors import KNeighborsClassifier 

# creating the data set 
from sklearn import  datasets 
import numpy as np 
X,y = datasets.make_classification(n_classes=2, n_features= 5, n_samples = 120, random_state=23) 


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 12) 


# creating the classifier 
clf = KNN(k =5)
clf.fit(X_train, y_train) 
predictions = clf.predict(X_test) 

accuracy = np.sum(predictions == y_test) / len(y_test) 
print(accuracy) 


