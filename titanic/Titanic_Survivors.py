#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 21:14:50 2019

@author: neeraj
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')
dataset["Embarked"][61]='S'
dataset["Embarked"][829]='S'
X = dataset.iloc[:, 2:11]
Test_X = test_dataset.iloc[:, 1:10]
y = dataset.iloc[:, 1].values

X = X.drop("Name",1)
Test_X = Test_X.drop("Name",1)

X = X.drop("Ticket",1)
Test_X = Test_X.drop("Ticket",1)

X_frame = X.drop("Cabin",1)
Test_X_frame = Test_X.drop("Cabin",1)

X = X_frame.values
Test_X = Test_X_frame.values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lab_enc = LabelEncoder()
X[:, 0] = lab_enc.fit_transform(X[:, 0]) 
X[:, 1] = lab_enc.fit_transform(X[:, 1]) 
Test_X[:, 0] = lab_enc.fit_transform(Test_X[:, 0]) 
Test_X[:, 1] = lab_enc.fit_transform(Test_X[:, 1]) 

temp = dataset.iloc[:, 11].values
temp = lab_enc.fit_transform(temp)
Test_temp = test_dataset.iloc[:, 10].values
Test_temp = lab_enc.fit_transform(Test_temp)

temp2 = np.zeros((891,7))
temp2[:, 0:6] = X
temp2[:, 6] = temp
Test_temp2 = np.zeros((418,7))
Test_temp2[:, 0:6] = Test_X
Test_temp2[:, 6] = Test_temp

X = temp2
Test_X = Test_temp2


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:7])
X = imputer.transform(X[:, 0:7])

imputer = imputer.fit(Test_X[:, 0:7])
Test_X = imputer.transform(Test_X[:, 0:7])



temp2 = np.zeros((891,8))
temp2[:, 0:7] = X
temp2[:, 7] = X[:, 3] + X[:, 4]
X = temp2

Test_temp2 = np.zeros((418,8))
Test_temp2[:, 0:7] = Test_X
Test_temp2[:, 7] = Test_X[:, 3] + Test_X[:, 4]
Test_X = Test_temp2



one_h_enc = OneHotEncoder(categorical_features=[0,6])
X = one_h_enc.fit_transform(X).toarray()
X = X[:, 1:]

one_h_enc = OneHotEncoder(categorical_features=[0,6])
Test_X = one_h_enc.fit_transform(Test_X).toarray()
Test_X = Test_X[:, 1:]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=22)

#Model 1
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators=300, criterion='gini', random_state=48)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
Test_y_pred1 = model1.predict(Test_X)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred1)

#Model 2
from sklearn.linear_model import LogisticRegression
classifier2 = LogisticRegression(random_state = 53)
classifier2.fit(X_train, y_train)
y_pred2 = classifier2.predict(X_test)
Test_y_pred2 = classifier2.predict(Test_X)
cm2 = confusion_matrix(y_test, y_pred2)

#Model 3
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_trainS = sc.fit_transform(X_train)
X_testS = sc.transform(X_test)
Test_X_Scaled = sc.transform(Test_X)
from sklearn.svm import SVC
classifier3 = SVC(kernel = 'rbf', random_state = 2)
classifier3.fit(X_trainS, y_train)
y_pred3 = classifier3.predict(X_testS)
Test_y_pred3 = classifier3.predict(Test_X_Scaled)
cm3 = confusion_matrix(y_test, y_pred3)

prediction1={}
prediction1["PassengerId"] = test_dataset["PassengerId"]
prediction1["Survived"] = Test_y_pred2
pd.DataFrame(prediction1, columns=["PassengerId",'Survived']).to_csv('Predictions1.csv', index=False)

prediction2={}
prediction2["PassengerId"] = test_dataset["PassengerId"]
prediction2["Survived"] = Test_y_pred3
pd.DataFrame(prediction1, columns=["PassengerId",'Survived']).to_csv('Predictions2.csv', index=False)
fun = pd.read_csv('Predictions2.csv')

prediction2 = pd.DataFrame(Test_y_pred3,columns=['Survived']).to_csv('Predictions2.csv')
"""objects = ['Class 1', 'Class 2', 'Class 3']
y_pos = np.arange(len(objects))
n1 = np.sum((Sur_P_ticket_fare[:, 1]==1)*Sur_P_ticket_fare[:, 0])
n2 = np.sum((Sur_P_ticket_fare[:, 1]==2) *Sur_P_ticket_fare[:, 0])
n3 = np.sum((Sur_P_ticket_fare[:, 1]==3) *Sur_P_ticket_fare[:, 0])

plt.bar(y_pos, [n1, n2, n3], alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Survived')
plt.title("Survival vs Passenger Class")"""