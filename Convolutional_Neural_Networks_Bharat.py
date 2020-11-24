# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:29:00 2020

@author: P795864
"""

#Convolutional Neural Networks

# Importing the libraries

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__


#PreProcessing the Training Set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
        
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
        
#Preprocessing the Test Set
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
        
#Building the CNN

#Initialising the CNN
cnn = tf.keras.models.Sequential()

#Step1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = [64,64,3]))

#Step2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

#Adding a Second Concolutional Layer    
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))#, input_shape = [64,64,3])) ---- As this input parameter will hv to apply only in first layer as it is connected with input
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))       

#Step3 - Flattening
cnn.add(tf.keras.layers.Flatten())

#Step4 - Full Connection
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))#---Units is number of neurons to add

#Step5 - Output Layer
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))#---Units is 1 because we are doing binary classification


#Training the CNN

#Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

#Training the CNN on training set and Eveluating it on the Test Set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


#Making a Single Prediction
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64,64)) #---Loading the image and have to mention the image size which we have declared earlier in trainig_set and test set which is 64 by 64
test_image = image.img_to_array(test_image)#-----As image are in PIL format and predict method only understands array, so we are converting image into array
test_image = np.expand_dims(test_image, axis = 0)#------As CNN processing the image into batches which is mentioned earlier, so we have to expand the dimensions for the single image and write the axis as 0 whihc means first batch
result = cnn.predict(test_image)
training_set.class_indices #--------this will tell us about the indices of dog and cat which is 0 and which is 1
if result[0][0] == 1: #----[0][0] indicates it will take the batch 0 of first image in that batch as pythin starts with 0
    prediction = 'dog'
else:
    prediction = 'cat'
    
print(prediction)    



