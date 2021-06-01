



# lda goal: 
# project data set onto a lowe dimensional space with good class -seperability 
# this is a supervised learning 

# pca goal : 
# finding the component axes that maximize the variance of the data : unsupervised. 


import numpy as np 

class LDA:
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None 

    def fit(self,X,y):
        n_features = X.shape[1] 
        class_labels = np.unique(y) 

        # s_w, sb 
        mean_overall  = np.mean(X, axis = 0)
        s_w = np.zeros((n_features, n_features) ) 
        s_b = np.zeros ((n_features, n_features))


        for c in class_labels:
            # only the samples of this class 
            x_c = X[y ==c]
            mean_c = np.mean(x_c, axis = 0)

            # (4, n_c) * (n_c , 4) = (4,4)
            s_w += (x_c -mean_c).T.dot(x_c - mean_c) 

            n_c = x_c.shape[0] # no of samples 
            mean_diff = (mean_c - mean_overall).reshape(n_features,1) 
            s_b += n_c * (mean_diff).dot(mean_diff.T)

        A = np.linalg.inv(s_w).dot(s_b) 
        eigenvalues, eigenvectors = np.linalg.eig(A) 
        eigenvectors = eigenvectors.T 
        indxs  = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[indxs]
        eigenvectors = eigenvectors[indxs] 

        self.linear_discriminants = eigenvectors[0:self.n_components] 
    def transform(self, X):
        return np.dot(X,self.linear_discriminants.T) 

    


from sklearn import datasets 
import numpy as np 


X,y= datasets.load_iris(return_X_y=True) 
lda = LDA(2)
lda.fit(X,y) 
X_projected = lda.transform(X) 

print(X.shape)
print(X_projected.shape) 

