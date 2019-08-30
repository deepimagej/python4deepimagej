# Important librairies

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K
import tensorflow as tf

# -----------------------------------------------------------------------------

def binary_crossentropy_weighted(weights):
    """
    Custom binary cross entropy loss. The weights are used to multiply
    the results of the usual cross-entropy loss in order to give more weight
    to areas between cells close to one another.
    
    The variable 'weights' refers to input weight-maps.
    """
    
    def loss(y_true, y_pred): 
        
        return K.mean(weights * K.binary_crossentropy(y_true, y_pred), axis=-1)
    
    return loss

# -----------------------------------------------------------------------------

def unet_weights(input_size = (256,256,1), learning_rate = 1e-4, weight_decay = 5e-7):
    """
    Weighted U-net architecture.
    
    The tuple 'input_size' corresponds to the size of the input images and labels.
    Default value set to (256, 256, 1) (input images size is 256x256).
    
    The float 'learning_rate' corresponds to the learning rate value for the training.
    Defaut value set to 1e-4.
    
    The float 'weight_decay' corresponds to the weight decay value for the training.
    Default value set to 5e-7.
    """
    
    # Get input.
    input_img = Input(input_size)
    
    # Get weights.
    weights = Input(input_size)
    
    # Layer 1.
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_img)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Layer 2.
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Layer 3.
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Layer 4.
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # layer 5.
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Layer 6.
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    # Layer 7.
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    # Layer 8.
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    # Layer 9.
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    # Final layer (output).
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    # Specify input (image + weights) and output.
    model = Model(inputs = [input_img, weights], outputs = conv10)

    # Use Adam optimizer, custom weighted binary cross-entropy loss and specify metrics
    # Also use weights inside the loss function.
    model.compile(optimizer = Adam(lr = learning_rate, decay = weight_decay), loss = binary_crossentropy_weighted(weights), metrics = ['accuracy'])

    return model


