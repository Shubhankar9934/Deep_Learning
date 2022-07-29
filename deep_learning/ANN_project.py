

# Artificial Neural Network


# Data Preprocessing

# Importing the Libraries
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset

dataset = pd.read_csv('C:\\Users\\shubh\\Desktop\\Deep Learning\\deep_learning\\Churn_Modelling.csv')
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

# Importing The Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

# Intialiation Of ANN

classifier = Sequential()

# Adding the input layer and first hidden layer
classifier.add(Dense(units= 6,kernel_initializer='he_uniform',activation='relu',input_dim= 11))
#classifier.add(Dropout(0.3))
# Adding the Second Hidden Layer
classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu'))
#classifier.add(Dropout(0.2))
# Adding the output layer

classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='Adamax',loss = 'binary_crossentropy',metrics = ['accuracy'])

# fitting the ANN to training set
model_history = classifier.fit(X_train,y_train,batch_size= 30,epochs=100,validation_split=0.33)

# list all data in history

print(model_history.history.keys())

# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)

score

