# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:08:43 2020

@author: P795864
"""

#Artificial Neural Networks

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values


# Encoding categorical data

# Label Encoding the Gender Column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

print(X)

#One Hot Encoding the Geography Column
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

print(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling is very important for Deep learning(Its a necessary step)
print(X_train)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#Building the ANN

#Initializing the ANN
ann = tf.keras.models.Sequential()

#Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))#---Units is number of neurons to add

#Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

#Adding the Output Layer------ Units = 1 as output is binary i.e one dimensional
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))#---Units is 1 because we are doing binary classification
#we are using activation = sigmoide bcz we are computing binary results, 
#but if its not binary and categorical that means more than one result then activation = softmax

#Training the ANN

#compiling the ANN ----
#Optimizer - To update the weights and reduce the loss error between predictions and real result (we use adam optimiser for Gradient Descent)
#loss - which is weigh to compute the difference between the real results and predictions (For binary results Loss is 'binary_crossentropy' and for categorical(more than one results) its 'categorical_crossentropy')
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Training the ANN on the training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)


#Making the Predictions and Evaluatinng the Model

"""
Homework Challenge
Use our ANN Model to predict if the customer with the following informations will leave the bank
Geography : France
Credit Score : 600
Gender : Male
Age : 40 years old
Tenure : 3yrs
Balance : $60000
Number of products : 2
Does this customer have a credit card? Yes
Is this customer a active member. yes
Estimated Salary : $50000
So, Should we say goodbye to this customer
"""

#Solution
#For Predict method always use 2-D arrays [[]] which is mandatory....... and always do scaling as its mandatory for ANN
print(ann.predict(sc_X.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))

#Predicting the Test Set Results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)),1))

#Making the COnfusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
