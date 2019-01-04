# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

 #step 1. import data
delim = "\t"
cols = ['Frequency', 'AngleOfAttack', 'ChordLength', 'FreeStreamVelocity', 'SuctionSideDisplacementThickness', 'ScaledSoundPressureLevel']
df = pd.read_csv('E:\\ML class\\airfoil_self_noise.dat.txt', delimiter=delim, names=cols)
print(df.head())
df.columns = ['Frequency', 'AngleOfAttack', 'ChordLength', 'FreeStreamVelocity', 'SuctionSideDisplacementThickness', 'ScaledSoundPressureLevel']
#features = cols[:len(cols) - 1]  # df.columns [:len(df.columns) - 1]
#or
features = ['Frequency', 'AngleOfAttack', 'ChordLength', 'FreeStreamVelocity', 'SuctionSideDisplacementThickness']
y_label = ['ScaledSoundPressureLevel']
x_df = df[features]
y_df = df[y_label]



# step 2. normalize the x data.
x_df = (x_df - x_df.mean()) / x_df.std()
print(x_df.head())



#step3. create a matrix for hyperparabola

#setting the matrixes

x_data_set = x_df.iloc[:,:-1].values
y_data_set = y_df.iloc[:,0].values

print(x_data_set,y_data_set)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data_set, y_data_set, test_size =0.35, shuffle=False)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)