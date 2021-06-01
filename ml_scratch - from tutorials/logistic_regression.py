


import numpy as np 

class LogisticRegression():
    def __init__(self, learning_rate = 0.001, epoch = 1000):
        self.learning_rate = learning_rate 
        self.epoch = epoch 
        self.weights = None 
        self.bias = None 

    
    # creating the sigmoid function to transfom the values between 0 and 1 to classes
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x)) 

    def fit (self,X,y):

        # getting the sample size and feature size 
        n_samples, n_features = X.shape

        # initializing the weights adn bias 
        self.weights = np.zeros(n_features)
        self.bias  = 0 

        # looping over the epochs 
        for _ in range(self.epoch):
            linear_result = np.dot(X,self.weights) + self.bias 

            # applying the signoid function 
            y_predicted = self.sigmoid(linear_result)

            # computig the derivative 

            errors = y_predicted - y 
            dw = (1/n_samples) * np.dot(X.T, errors)  
            db = (1/n_samples) * np.sum(errors)

            self.weights -= dw*self.learning_rate
            self.bias -= db*self.learning_rate 


     
    def predict(self, X):
        linear_output = np.dot(X, self.weights)
        probs = self.sigmoid(linear_output) 
        predicted_class = [i if i> 0.5 else 0 for i in probs] 
        return predicted_class 

from sklearn.linear_model import LogisticRegression 

from logistic_regression import  LogisticRegression 
from sklearn import  datasets 
import numpy as np 
X,y = datasets.make_classification(n_classes=2, n_features= 5, n_samples = 120, random_state=23) 
# print(X)
# print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 12) 

Lmodel = LogisticRegression() 
Lmodel.fit(X_train, y_train)

predictions = Lmodel.predict(X_test) 

accuracy = np.mean(predictions == y_test)  
print(accuracy) 

