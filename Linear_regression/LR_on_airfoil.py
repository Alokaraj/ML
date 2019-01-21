# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

 #step 1. import data
delim = "\t"
df = pd.read_csv('airfoil_self_noise.txt', delimiter=delim)

print(df.head())


#step2. create a matrix for hyperparabola

#setting the matrixes

x = df.iloc[:,:5].values
y = df.iloc[:,-1].values

print(x.shape)
print(x)
print(y)

# step 3. normalize the x data.
x = (x - x.mean()) / x.std()
print(x)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size =0.35, shuffle=False)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)

