#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 19:48:20 2017

@author: krishna
"""

# Data Pre-processing

#Importing Libraries

"""
Following three libraries are essential 
in all machine models/scripts
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Best Library avaliable to import and manage datasets #

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

# Handling missing Data
from sklearn.preprocessing import Imputer #Imputer is a class we are importing, class needs an object to act on
#Below varibale is an object for the class imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
imputer = imputer.fit(X[:,1:3])
#mean values need to be placed in missing field of the column. Use the below the statement for to accomplish
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#creating an object for labelencoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
# Read Dummy Encoding using One hot Encoder
onehotencoder = OneHotEncoder(categorical_features = [0]) #Column Index that needs to be treated as category
# Country Column is replaced by three columns
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y= LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# Object sc_X should be first fitted to training model and then fitted to 
# test model so that both are scaled on the same basis
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
