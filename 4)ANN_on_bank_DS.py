# Artificial Neural Network
#predicting the customer's gonna leave the bank or not according to the dataset

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')


X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values
print(X)
print(y)


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])#which reats the dummy variable for index 1, so that it should not compaire spain france , germany as higher or lower
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]#removing index 0(first column), dummy variable
#print(X)
 


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras #uses tensorflow backend
from keras.models import Sequential #require to initilize ANN
from keras.layers import Dense #require to build layers of ANN

# Initialising the ANN (2 types 1.defining the sequence of layers. 2.defining a graph)
#v r using type 1
classifier = Sequential() #object creation.

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))#here no of input layer is(independent variables) (input_dim)11 and no of output layer is 1 so 11+1/2=6 (outpit_dim=6)is the hidden first layers,init parameter used to initilize the weights, relu is an rectifire function, v use it in all hiffen layers, except output layer there v use sigmoid function.
#or (using parameter tuning v can tune the no of hidden layers)

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))#here no of input layer is 11 and no of output layer is 1 so 11+1/2=6 is the hidden first layers, relu is an rectifire function, v use it in all hiffen layers, except output layer there v use sigmoid function.

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))#here output_dim depends on no of catogories, v should have probability so v use sigmoid, if v had 2 to 3 catogary output v should use softmax function.

# Compiling the ANN(over here v r going to apply socastic gradient decent) 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])#compile is a method.1st parameter is adam is a socastic gradient decent method.2nd parameter is logarithmic lass function ,creating list for accuracy.
# if our dependent variable has binary outcome it is called binary_crossentropy, if our dependent variable has more than 2 outcomes like 3 category it is called categorical_crossentropy.

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100) #batch_size=no of observations after v want to update the weights.




# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #taking threshold 0.f to make true or false.
print(y_pred)


# Making the Confusion Matrix and classification report
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)

cr = classification_report(y_test, y_pred)
print(cr)
