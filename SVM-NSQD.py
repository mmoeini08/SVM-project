# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:43:05 2021

@author: mmoein2
"""

#Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler #New element this week!
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score #New element this week! K-cross validation
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name = 'TSS'
data = pd.read_csv(file_name + '.csv', header=0, index_col=0) #Remove index_col = 0 if rows do not have headers

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")

#Defining X1, X2, and Y
X1 = 'DA'
X2 = 'PR1'
X3 = 'PI1'
X4 = 'PC'
X5 = 'PI'
X6 = 'PO'
X7 = 'PF'
X8 = 'IP'
X9 = 'AD'
X10 = 'PR'
X11 = 'RO'
Y1 = 'TSS'
X = np.column_stack((data[X1], data[X2], data[X3], data[X4], data[X5], data[X6], data[X7], data[X8], data[X9], data[X10], data[X11]))

Y = data[["TSS"]]

#Using Built in train test split function in sklearn
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

######################################################
#Emphasis on the need to scale data to increase performance.
#And it is good practice to scale after train_test_split to avoid any data snooping issues.
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
######################################################

#Setting up SVM and fitting the model with training data
ker = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'] #we won't use precomputed
vector = svm.SVR(kernel=ker[3], C=1.1, gamma=0.07, degree=9, epsilon=0.0001) #degree is only relevant for the 'poly' kernel, and gamma is relevant for the 'rbf', 'poly', and 'sigmoid' kernels, default is 'scale'

#Cross Validation (CV) process
scores = cross_val_score(vector, X_train, Y_train, cv=5)
print(scores)
print("Accuracy: {0} (+/- {1})".format(scores.mean().round(2), (scores.std()*2).round(2)))
print("")

#Fit final SVC
print("Now fit on entire training set")
print("")
vector.fit(X_train, Y_train)

print("Indices of support vectors:")
print(vector.support_)
print("")
print("Support vectors:")
print(vector.support_vectors_)
print("")
print("Intercept:")
print(vector.intercept_)
print("")

if vector.kernel == 'linear':
    c = vector.coef_
    print("Coefficients:")
    print(c)
    print("This means the linear equation is: y = -" + str(c[0][0].round(2)) + "/" + str(c[0][1].round(2)) + "*x + " + str(vector.intercept_[0].round(2)) + "/" + str(c[0][1].round(2)))
    print("")

#Run the model on the test (remaining) data and show accuracy
Y_predict = vector.predict(X_test)
print(Y_predict)
print(Y_test.values)
print(metrics.r2_score(Y_predict, Y_test))


