# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## STEP 1:
Import the required packages and print the present data.
## STEP 2:
Find the null and duplicate values.
## STEP 3:
Using logistic regression find the predicted values of accuracy , confusion matrices.
## STEP 4:
Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: RIYA P L
RegisterNumber: 212223240141

import pandas as pd
data =pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
lr =LogisticRegression(solver ="liblinear")
lr.fit(x_train,y_train)
ypred=lr.predict(x_test)
ypred
from sklearn.metrics import accuracy_score, classification_report
accuracy= accuracy_score(y_test, ypred)
accuracy
classification_report1= classification_report(y_test, ypred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:

![image](https://github.com/user-attachments/assets/0b61ac5e-f296-48f8-b041-a37960c530e7)

![image](https://github.com/user-attachments/assets/f67e186d-bc26-4d3e-85de-85715f7ca0a5)

![image](https://github.com/user-attachments/assets/ab47b304-9703-4fe1-a013-829012a800fb)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
