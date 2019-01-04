#The MNIST database contains 60,000 training images and 10,000 testing images.
#x_train and x_test parts contain greyscale RGB codes (from 0 to 255) while y_train and y_test parts contains labels from 0 to 9 which represents which number they actually are.



import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


import matplotlib.pyplot as plt
#%matplotlib inline # Only use this if using iPython
image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8
plt.imshow(x_train[image_index], cmap='Greys')


#We need to know the shape of the dataset to channel it to the convolutional neural network.
print(x_train.shape)# 60000 images in train dataset and (28, 28) represents the size of the image: 28 x 28 pixel.



# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
#input_shape = (28, 28, 1)#28 x 28 image format and 1 channel

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])



# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28,3,3, input_shape= (28, 28, 1), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Adding a second convolutional layer #to make more optimized.
model.add(Conv2D(28,3,3, activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # Flattening the 2D arrays for fully connected layers

#creating ANN
model.add(Dense(output_dim = 128, activation ='relu')) #128 is experimented value.
model.add(Dropout(0.2))# overcome overfitting.
model.add(Dense(output_dim = 64, activation ='relu'))#128/2=64
model.add(Dropout(0.2))
model.add(Dense(output_dim = 10, activation ='softmax'))

# Compiling the CNN
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#fit CNN
model.fit(x=x_train,y=y_train, epochs=10)

#evaluate the model.
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) #Test accuracy: 0.9960


'''
# individual predictions
image_index = 4444
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1))
print(pred.argmax())'''

