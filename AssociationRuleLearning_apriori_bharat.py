# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:27:29 2020

@author: P795864
"""

#Apriori

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset with pandas
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
#Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
#min_suport = we have assumed that item has been purchased 3 times a day and based on that 
#we hv calculated support
#min-support = 3(day)*7(week)/7500 = 0.0028

#min_confidence = Person will definitely buy that combination ...... we hv set it up 20%

#min_lift =  Ususally lift is 4,5 or 6.....but we r considering as 3 .....any value greater tha 3 its 
#high chance that combination is good

#Visualising the results
results = list(rules)

