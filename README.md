# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries (e.g., pandas, numpy,matplotlib).
2. Load the dataset and then split the dataset into training and testing sets using sklearn library.
3. Create a Linear Regression model and train the model using the training data (study hours as input, marks scored as output).
4. Use the trained model to predict marks based on study hours in the test dataset.
5. Plot the regression line on a scatter plot to visualize the relationship between study hours and marks scored.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Shehan Shajahan
RegisterNumber: 212223240154
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/MLSET.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='orange')
lr.coef_
lr.intercept_
```

## Output:
## 1) Head:
![out1](https://github.com/shehanshajahan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139317389/1a913076-617e-4243-beb7-5bea99de59dd)
## 2) Graph Of Plotted Data:
![out2](https://github.com/shehanshajahan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139317389/09c4defd-f873-4436-bc90-7da8ab0ded1b)
## 3) Trained Data:
![out3](https://github.com/shehanshajahan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139317389/b5f943eb-f829-4573-b7e8-60034f59afed)
## 4) Line Of Regression:
![out4](https://github.com/shehanshajahan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139317389/a1c93f38-125a-48ac-b4c8-9b3cf27598ea)
## 5) Coefficient And Intercept Values:
![out5](https://github.com/shehanshajahan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139317389/0d2d2a50-d3d3-40e3-9697-6735fd3288d9)

 
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
