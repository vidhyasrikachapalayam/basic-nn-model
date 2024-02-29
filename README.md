## EX1 Developing a Neural Network Regression Model


## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

Name:VIDHYASRI.K
Register Number: 212222230170

## DEPENDENCIES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

## DATA from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('DL').sheet1

rows = worksheet.get_all_values()
## DATA PROCESSING
df = pd.DataFrame(rows[1:], columns=rows[0])
df.head()

df['Input']=pd.to_numeric(df['Input'])
df['Output']=pd.to_numeric(df['Output'])

X = df[['Input']].values
y = df[['Output']].values

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)

Scaler = MinMaxScaler()
Scaler.fit(x_train)

x_train.shape
x_train1 = Scaler.transform(x_train)
x_train1.shape
## MODEL ARCHITECTURE AND TRAINING
model = Sequential([
    Dense(units = 5,activation = 'relu',input_shape=[1]),
    Dense(units = 2, activation = 'relu'),
    Dense(units = 1)
])

model.compile(optimizer='rmsprop', loss = 'mae')

model.fit(x_train1,y_train,epochs = 2000)

model.summary()
## LOSS CALCULATION
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
## PREDICTION
x_test1 = Scaler.transform(x_test)
model.evaluate(x_test1,y_test)

x_n = [[21]]
x_n1 = Scaler.transform(x_n)
model.predict(x_n1)
## DATASET INFORMATION
![image](https://github.com/vidhyasrikachapalayam/basic-nn-model/assets/119477817/a350d6f9-1f94-4cf1-b817-186cea31acef)



### Training Loss Vs Iteration Plot

![image](https://github.com/vidhyasrikachapalayam/basic-nn-model/assets/119477817/33926c94-230f-4be3-aa6e-8dd095a2d06e)


### Test Data Root Mean Squared Error

![image](https://github.com/vidhyasrikachapalayam/basic-nn-model/assets/119477817/b8cf3967-a365-45c6-a843-3e9eff059618)


### New Sample Data Prediction
![image](https://github.com/vidhyasrikachapalayam/basic-nn-model/assets/119477817/27cb5ec5-98b2-4dff-a4d2-26069f623fa0)


## RESULT
Summarize the overall performance of the model based on the evaluation metrics obtained from testing data as a regressive neural network based prediction has been obtained.
