# Convolutional Neural Network
# to predict cat or not.8000 training set, 2000 test set.


# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential #to initilize neural networks
from keras.layers import Convolution2D # IMAGES HAVE 2D, VIDEOS HAVE 3D.
from keras.layers import MaxPooling2D 
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN (4 steps)
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu')) 
#Convolution2D's argument 1. No of feature detectors, 2. no of columns of it, 3. no of rows of it.(border_mode='same') its a default argument, 
#input_shape=(64 x 64 image format and 3 channels, couse its color image, v can use a larger format in GPUs 128*128 and 256*256)
# hhere v r using tensorflow backend order in input_shape , theano backend has different order. here v use relu= to remove -ve values, to get non-linearity. 

# Step 2 - Pooling(size of the feature map divided by 2)to reduce no of features.
classifier.add(MaxPooling2D(pool_size = (2, 2))) #pool_size=2x2dimentions.

# Adding a second convolutional layer() #to make more optimized.
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten()) #v get 1d vector.

# Step 4 - Full connection.
classifier.add(Dense(output_dim = 128, activation = 'relu'))
#classifier.add(Dense(output_dim = 64, activation = 'relu')) #128/2= 64 #should check with this(DOUBT)

classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) #for more than binary outcomes v use softmax.


# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Part 2 - Fitting the CNN to the images

#(go to keras documentation, process called image augumentation it consists of pre processing d image to prevent overfitting,it performs some randome transformations on images.)

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
								#rescale (corespond to feature scalling part of data preprocessing) makes all values lie between 0&1,shear_range its a geometrical transformation where d pixals are moved to fixed directions over a proportional distance from d line that is parallel to direction they r moving to,random zoom,flip horizontally.  

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
												 #target_size as mentioned in Convolution2D layer input_shape= 64,64 dimentions.

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)