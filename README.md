# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1. Import necessary libraries.
 2. Load and inspect the dataset.
 3. Normalize the data.
 4. Implement the cost function and gradient descent manually.
 5. Train the model using the normalized data.
 6. Make a prediction using the trained model.
 7. Reverse scale to get the original prediction value.



## Program:
# Program to implement linear regression using gradient descent.
```
Developed by: T Ajay
Register Number: 212223230007
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]  # Add intercept
    theta = np.zeros(X.shape[1]).reshape(-1, 1)

    for _ in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= learning_rate * (1/len(X1)) * X.T.dot(errors)
    return theta
data = pd.read_csv('50_Startups.csv')
print("Dataset Preview:\n", data.head())
X = data.iloc[1:, :-2].astype(float).values 
y = data.iloc[1:, -1].astype(float).values.reshape(-1, 1)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)
theta = linear_regression(X_scaled, y_scaled)
new_data = np.array([[165349.2, 136897.8, 471784.1]])
new_data_scaled = scaler_X.transform(new_data)
prediction_scaled = np.dot(np.c_[np.ones((1, 1)), new_data_scaled], theta)
prediction = scaler_y.inverse_transform(prediction_scaled)
print(f"Predicted Profit: {prediction[0][0]}")
```
## Output:
![image](https://github.com/user-attachments/assets/0a709d39-5645-4188-b3a7-fec345a7fe86)

## Values of X and Y
![image](https://github.com/user-attachments/assets/9fe58d09-d0f6-4755-ae9b-17947d46504c)

![image](https://github.com/user-attachments/assets/59fa1388-0d37-4c5e-a5e4-2f5d10d109d3)

## Predicted Value
![image](https://github.com/user-attachments/assets/7ef530d8-587b-4eb7-8371-27a9621cb5a7)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
