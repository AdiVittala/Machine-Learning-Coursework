#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 20:28:15 2017

@author: krishna
"""

# Multiple Linear Regression

# First Step is to import data - so use the data preprocessing code

# Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:,1:] # Usually python varibales avoids one of the dummy varibale. So no need to include this. But you can.
# What the above statement does is, it selects all the columns from column 1 till the end.


#Splitting the dataset in to training set and Test Set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set Results
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
# Preparation for Backward elimination process

import statsmodels.formula.api as sm # Stats model library does not consider the constant b0 which is part of multiple regrssion expression
# Hence we need to add a column of 1 to our dataset
#X = np.append(arr = X, values = np.ones((50, 1)).astype(int), axis = 1) # This statement adds column of 50 ones at the end. But we want to add the column of ones at the beginning.
# To do this we change the above statement as follows
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS()