


# creating a linear regression class 
import numpy as np 
class linearRegression():
    def __init__(self, lr  = 0.001, n_iters = 1000):
        self.lr = lr 
        self.n_iters = n_iters 
        self.weights = None 
        self.bias = None 
        
    def fit(self, X,y):
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) 
        self.bias = 0 
        
        for _ in range(self.n_iters):
            
            linear_output = np.dot(X, self.weights) + self.bias 
            dw = (1/n_samples) + np.sum(np.dot(X.T, linear_output -y)) 
            db = (1/n_samples) + np.sum(linear_output -y) 
            
            self.weights -= self.lr * dw 
            self.bias = self.lr * db  
    def predict (self, X) :
        y_approximated = np.dot(X, self.weights) + self.bias 
        return y_approximated

# importing libraries 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# generting data set for regression using sklearn.datasets 
X, y = datasets.make_regression(n_samples=100, n_features=3, noise=20, random_state=4)
print(X)
print(X.shape) 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    
from sklearn.linear_model import LinearRegression 

# training the dat a
regressor = linearRegression() 
regressor.fit(X_train,y_train) 
predictions = regressor.predict(X_test)


# creating a fuction for printing the accuraies 
def accuracies(predictions, y_test):
    error = predictions -y_test
    ess = np.sum(error**2) 
    tss = np.sum((y_test - np.mean(y_test))**2) 
    r2 = 1 - (ess/tss) 
    print( 'mse  = ', np.mean((error)**2))
    print( 'mae = ', np.mean(abs(error)))
    print('r2 = ', r2)
 
 # printing the result 
accuracies(predictions, y_test)





from sklearn.datasets import  load_boston 
X,y = load_boston(return_X_y=True) 
# print(X) 
# print(X.shape) 
# print(y) 

