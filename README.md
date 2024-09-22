# Implementation-of-CNN

## AIM

To Develop a convolutional deep neural network for digit classification.

## Problem Statement and Dataset
The goal of this project is to develop a Convolutional Neural Network (CNN) to classify handwritten digits using the MNIST dataset. Handwritten digit classification is a fundamental task in image processing and machine learning, with various applications such as postal code recognition, bank check processing, and optical character recognition systems.

The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9), totaling 60,000 training images and 10,000 test images. The challenge is to train a deep learning model that accurately classifies the images into the corresponding digits.

## Neural Network Model

Include the neural network model diagram.(http://alexlenail.me/NN-SVG/index.html)

## DESIGN STEPS

### STEP 1:
Import the necessary libraries and Load the data set.

### STEP 2:
Reshape and normalize the data.

### STEP 3:
In the EarlyStoppingCallback change define the on_epoch_end funtion and define the necessary condition for accuracy

### step 4:
Train the model


## PROGRAM

### Name: Divakar R
### Register Number: 212222240026
```python

import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train=x_train.astype('float32') / 255.0
x_test=x_test.astype('float32')/255.0
x_train
x_test


x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
x_train
x_test

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.99:
            print("\nAccuracy reached 99%, stopping training.")
            self.model.stop_training = True

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Build the CNN model
model = Sequential()

# 1st Conv layer
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd Conv layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output to feed it into fully connected layers
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output layer for 10 classes
model.add(Dense(10, activation='softmax'))

!pip install tensorflow
import tensorflow as tf

# Assuming y_train contains integer labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with the custom callback
callback = CustomCallback()
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2, callbacks=[callback])
```

## OUTPUT

### Reshape and Normalize output

![Screenshot 2024-09-22 141040](https://github.com/user-attachments/assets/a70bc9d4-4322-4496-a7f4-37ec073de1f3)


### Training the model output

![image](https://github.com/user-attachments/assets/c3c77909-79eb-402a-9570-1e801dbb1081)



## RESULT
Thus the program for developing a convolutional deep neural network for digit classification.

