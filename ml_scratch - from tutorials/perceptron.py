

# creating a class for perceptron 

import numpy as np 

class Perceptron:
    def __init__(self, learning_rate = 0.001, epoch = 1000):
        self.learning_rate = learning_rate 
        self.epoch =epoch 
        self.activation = self.unit_step_func 
        self.weights =  None
        self.bias = None  
    
    def unit_step_func(self,x):
        return np.where(x>=0, 1, 0)  

    def fit(self,X,y):
        # getting the sample no and fetures counts 
        n_samples, n_features = X.shape 

        #initializing bias and weights 
        self.weights =  np.zeros(n_features) 
        self.bias  = 0 

        # redefining training y 
        y_  = np.array([1 if i>0 else 0 for i in y ])
 
        for _ in range(self.epoch):
            for idx,x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias 
                y_predicted = self.activation(linear_output)


                update = self.learning_rate * (y_[idx] - y_predicted) 
                self.weights += update * x_i 
                self.weights += update 

    def predict(self,X):
        linear_output  = np.dot(X,self.weights) + self.bias 
        y_predicted  = self.unit_step_func(linear_output) 
        return y_predicted 
    


from sklearn import  datasets 
import numpy as np 
X,y = datasets.make_classification(n_classes=2, n_features= 5, n_samples = 120, random_state=23) 
# print(X)
# print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 12) 

Lmodel = Perceptron()
Lmodel.fit(X_train, y_train)

predictions = Lmodel.predict(X_test) 

accuracy = np.mean(predictions == y_test)  
print(accuracy) 






from sklearn.linear_model import Perceptron 
from sklearn.datasets import load_digits 
F,t = load_digits(return_X_y=True) 
clf = Perceptron(random_state=12) 
clf.fit(F,t) 


print(F) 
print(F.shape)
print('------------------')
print(y)
print(y.shape)  
score = clf.score(F, t) 
print(score) 
