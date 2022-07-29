# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 23:23:44 2022

@author: shubh
"""

# Artificial Neural Network


# Data Preprocessing

# Importing the Libraries
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv(r'C:\\Users\\shubh\\Desktop\\Deep Learning\\deep_learning\\Churn_Modelling.csv')
X = dataset.iloc[: ,3:13]
y = dataset.iloc[:,13]

# Create Dummy Feature/Variables
geography = pd.get_dummies(data = X["Geography"],drop_first=True)
gender = pd.get_dummies( data= X["Gender"],drop_first=True)

# Concatenate the DataFrame

X = pd.concat([X,geography,gender],axis = 1)

## Drop Unnecessory Columns

X = X.drop(['Geography','Gender'],axis = 1)

# Splitting the Dataset into the training set and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train,y_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


## Perform Hyperparameter Optimization

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.activations import relu, sigmoid

# Creating Model(Hyperparameter Tuning)
def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=X_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
    model.add(Dense(units = 1, kernel_initializer= 'glorot_uniform', activation = 'sigmoid')) # Note: no activation beyond this point
    
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, verbose=0)


layers = [[20], [40, 20], [45, 30, 15]]
activations = ['sigmoid', 'relu']
param_grid = dict(layers=layers, activation=activations, batch_size = [128, 256], epochs=[30])
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)

grid_result = grid.fit(X_train, y_train)

[grid_result.best_score_,grid_result.best_params_]

