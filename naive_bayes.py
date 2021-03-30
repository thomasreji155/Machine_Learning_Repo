import pandas as pd 
import numpy as np 

class NB():
    def __init__(self,X,y):
        self.X = X
        self.y = y
        self.features = list(self.X.columns)
        
         # making a list of unique classes
        self.classes = list(self.y.unique())
        # counting the total no of classes in the target feature 
        self.class_0count = list(self.y).count(self.classes[0])
        self.class_1count = list(self.y).count(self.classes[1]) 
        
    def fit(self):       
        # getting the features names and target name      
        target = self.y.name
            
        # creating a dictionary with features as keys 
        final_dict = dict.fromkeys(self.features)

        # start looping over the features using feature index 
        for f in range(len(self.features)):
            
            # creating the gps list consists of uniques values for each feature 
            gps = list(self.X[self.features[f]].unique())
            
            # creating a dictionary for each gp  
            gp_counts = dict.fromkeys(gps)
            
            for j in range(len(gps)):          
                # getting the count of first group 
                dict_count = self.y[self.X[self.features[f]] == gps[j]].value_counts()
                
                # dividing the value counts by total no of classes respectively (yes and no)
                dc = dict.fromkeys(self.classes)
                dc[self.classes[0]] = dict_count[self.classes[0]]/self.class_0count
                dc[self.classes[1]] = dict_count[self.classes[1]]/self.class_1count
                
                # storing the counts dictionary in 
                gp_counts[gps[j]] = dc
                         
            # adding all the gp counts to the respective features :    
            final_dict[self.features[f]] = gp_counts
                    
        # returning the final dictionary which holds the apriori probabilities 
        return final_dict
    
    def predict(self):
        final_dict = NB.fit(self)
        # defining a function 
        def pred(feature_vec):                   
            # finding the class probabilities        
            class_proba = dict.fromkeys(self.classes)
            for i in range(len(self.classes)):
                ct = len(self.y[self.y == self.classes[i]])/len(self.y)
                class_proba[self.classes[i]] = ct            
            
            prob_yes = 1
            prob_no = 1
            for i in range(len(feature_vec)):
                r = final_dict[self.features[i]]  [feature_vec[i]]    [self.classes[0]]
                prob_yes = prob_yes * r  
                
                g = final_dict[self.features[i]][feature_vec[i]]    [self.classes[1]]
                prob_no = prob_no * g 
                

        
        
            # multiplying with class probabilities     
            fin_yes = prob_yes * class_proba[self.classes[0]]
            fin_no = prob_no * class_proba[self.classes[1]]
            
            if fin_yes> fin_no:
                return self.classes[0]
            else:
                return self.classes[1]
        
        preds = []
        for row in range(len(self.X)):
            preds.append(pred(list(self.X.iloc[row])))
        return pd.DataFrame(preds,columns = ['predictions'])  


