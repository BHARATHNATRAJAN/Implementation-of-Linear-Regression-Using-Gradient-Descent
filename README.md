# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Intialize weights randomly.
2.Compute predicted.
3.Compute gradient of loss function.
4.Update weights using gradient descent.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: N.BHARATH
RegisterNumber: 212223230030
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors = (predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
    
data=pd.read_csv('50_Startups.csv',header=None)
data.head()
X = (data.iloc[1:,:-2].values)
print(X)

X1=X.astype(float)
scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)

X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

theta = linear_regression(X1_Scaled,Y1_Scaled)

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1,new_Scaled),theta)
prediction = prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value:{pre}")

```
## Output:


![image](https://github.com/23004513/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138973069/cb36a87c-e1c2-4ee2-918b-582d5f5f0289)

![image](https://github.com/23004513/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138973069/92739baf-77b7-4122-944c-094a3cd84dcd)

![image](https://github.com/23004513/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138973069/5b12c991-1b1d-40f3-9af9-02cd56dd54a0)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
