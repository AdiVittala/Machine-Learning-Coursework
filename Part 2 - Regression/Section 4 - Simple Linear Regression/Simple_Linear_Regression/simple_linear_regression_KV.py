#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:20:26 2017

@author: krishna
"""

# Simple Linear Regression

# First Step is to import data - so use the data preprocessing code

# Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Splitting the dataset in to training set and Test Set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression # LinearRegression is a class in the module linear module
# Creating an object for the class
regressor = LinearRegression()
#calling a method on the object created
regressor.fit(X_train, y_train) # In terms of machine learning, regressor is the machine that is the learning the co-relation between the X_train and Y_train data

#Predicting the Test Set results
y_pred = regressor.predict(X_test) # Creating a vector to generate predicted values

# Visualizing the training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()