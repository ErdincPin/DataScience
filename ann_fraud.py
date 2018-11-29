# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:21:02 2018

@author: erdinc.pinar
"""
# import necessary libraries
import numpy as np
import pandas as pd

#import dataset
dataset = pd.read_csv('creditcard.csv')
X = dataset.iloc[:, 1:29].values
y = dataset.iloc[:, 30].values

#split dataset into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Compile and fit ANN

# import keras libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

# initialize ANN and add layers
classifier = Sequential()
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 28))
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#compile classifier
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the ANN classifier to training set
classifier.fit(X_train, y_train, batch_size = 100, epochs = 10)

#test set performance
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
