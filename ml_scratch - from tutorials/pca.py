import numpy as np 

class PCA:
    def __init__(self, n_components) :
        self.n_components = n_components
        self.components = None 
        self.mean = None 

    def fit(self, X):
        
        # mean 
        self.mean = np.mean(X,axis = 0) 
        X = X - self.mean 
        # covariance matrix 
        cov = np.cov(X.T)
        # eigen vectors and values 
        eigen_values, eigen_vectors = np.linalg.eig(cov) 
        # v[:,1] 
        # sorting the eigen vectors in decreasing order 
        eigen_vectors = eigen_vectors.T 
       #  print(eigen_vectors) 

        idxs = np.argsort(eigen_values)[::-1] 
        eigen_values = eigen_values[idxs] 
        eigen_vectors = eigen_vectors[idxs] 

        # storing only first n eigen vectors 
        self.components  = eigen_vectors[0:self.n_components]

        # printing the components 
        # print(self.components) 

    def transform(self, X):
        
        # project the data 
        X = X -self.mean 

        # print('\noriginal data'.center(50,'*'))
        # print(X) 
        
        return np.dot(X, self.components.T)  



X =  np.array([[1,5,3],[3,5,3],[7,5,3], [34,46,24],[43,63,2]])
# print(x) 
# print(x.shape)
# 
# cov = np.cov(x.T)  
# print(cov) 

pca = PCA(2) 
pca.fit(X) 
# print(pca.transform(X)) 


print(pca.components) 



# using sklearn 
from sklearn.decomposition import PCA 
PC = PCA(n_components=2) 
PC.fit(X) 
transformed_x = PC.fit_transform(X)
# print('\n\n Transformed data')  
# print(transformed_x)
print(PC.components_)

# when we have a test data for transforming the data we take the 
# dot product of components received from training and the test data 

test = np.dot(test, PC.components_.T) 
