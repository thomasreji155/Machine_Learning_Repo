import numpy as np 
import random 
import pandas as pd 

random.seed(23)
data = pd.DataFrame()
for i in range(3):
    data[i] = random.sample(range(10,100), 50)

print(data)


