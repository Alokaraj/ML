# data preprocessing

#import the libraries
#a library is a tool used to make specfic job
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('Data.csv' )
#print(dataset)



#create matrix of features
x = dataset.iloc[:,:-1].values #in iloc[] first we take all the lines(rows) and secondly v take all d columns excepy last one, i,e; last one is y
y = dataset.iloc[:,3].values # in iloc[] first we take all the lines and secondly v take the index of d y i,e; index starts from 0

print(x)
print(y)

#taking care of missing data
from sklearn.preprocessing import Imputer # imputer is a class
imputer = Imputer(missing_values = 'NaN',strategy='mean',axis=0) #here v r inspecting d class parameters first parameter is missing values, second is strategy, 3rd is axis if axis=1 means rows, if axis =0 means columns.
imputer = imputer.fit(x[:,1:3]) #we r only going to fit for missing data.
x[:,1:3] = imputer.transform(x[:,1:3])# we replace the missing data with there mean of the column
print(x)

#taking care of catogorical data.(words) data to numeric data.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder() # creating object of the LabelEncoder class.
x[:, 0] = labelencoder_x.fit_transform(x[:,0]) #we will get all the words in that perticular column will be encoded into numeric values.
# #the problem v get is it starts compare each other like if v have germany and france.
# #france is > germany like that so we create dummy values, like extra columns v create, using 0s and 1s.
onehotencoder = OneHotEncoder(categorical_features=[0]) # creating object of the OneHotEncoder class # we r going to have dummy variables # v r going to specify the index.
x = onehotencoder.fit_transform(x).toarray()
print(x)

# so ve have catogorical y value.
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(y)


#taking care of catogorical data.(words) data to numeric data.(espetally for decision trees)
d = {'Y':1 ,'N':0, 'BS':0, 'MS':1, 'PhD':2} #create a dictionary and assign the values and map them to pirticular columns.
dataset['Employed?'] = dataset['Employed?'].map(d)
dataset['Level of Education'] = dataset['Level of Education'].map(d)
dataset['Top-tier school'] = dataset['Top-tier school'].map(d)
dataset['Interned'] = dataset['Interned'].map(d)
dataset['Hired'] = dataset['Hired'].map(d)
print(dataset)




#splitting dataset into traing set and test set
from sklearn.model_selection import train_test_split #importing the library.
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 0)# parameters are arrays, test size ex= .5=50%. and random_state = 0.
print(x_train)


#feature scalling(here we do standardisation and Normalization)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)# v dont need to fit the test set because it is already fit into training set.
#question: do v need to scall the dummy values?
# Ans: it depends on the context

# for y_test and y_train v have to do feature scalling if it is regression.
# for catogorical variable of classification problem v shouln't do it.

#or for simple feature scalling(#i prefer)
dataset = (dataset - dataset.mean()) / dataset.std()


