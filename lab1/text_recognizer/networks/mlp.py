from typing import Tuple

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization, Lambda


def mlp(input_shape: Tuple[int, ...], # [28,28]
        output_shape: Tuple[int, ...], # (80,)
        layer_size: int=128,
        dropout_amount: float=0.2,
        num_layers: int=3) -> Model:
    """
    Simple multi-layer perceptron: just fully-connected layers with dropout between them, with softmax predictions.
    Creates num_layers layers.
    """
    num_classes = output_shape[0]

    model = Sequential()
    # Don't forget to pass input_shape to the first layer of the model
    ##### Your code below (Lab 1)
#    model.add(Flatten(input_shape=input_shape))
#    for _ in range(num_layers):
#        model.add(Dense(layer_size, activation= 'relu'))
#        model.add(Dropout(dropout_amount))
#    model.add(Dense(num_classes,activation='softmax'))
    ##### Your code above (Lab 1)
    
    ##### CNN Network #2: 
    
    # need to add an extra dimension to the input, so that I have 4D. 
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape=input_shape))
        input_shape = (input_shape[0], input_shape[1], 1)
           
    model.add(Conv2D(32, (3, 3), input_shape=(28,28,1))) # this applies 32 convolution filters of size 3x3 each. 
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    BatchNormalization(axis=-1)
    model.add(Conv2D(64,(3, 3)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    # Fully connected layer

    BatchNormalization()
    model.add(Dense(512))
    model.add(Activation('relu'))
    BatchNormalization()
    model.add(Dropout(dropout_amount))
        
    model.add(Dense(num_classes,activation='softmax')) 
    
    return model

