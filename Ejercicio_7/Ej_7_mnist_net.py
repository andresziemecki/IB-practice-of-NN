#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:57:09 2018

@author: andres
"""

from __future__ import print_function


import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os
import matplotlib.pyplot as plt
from Andy_Functions import load_data




# input image dimensions
img_rows, img_cols = 128, 128 # ponerlas tambien en la funcion load_data


batch_size = 32
num_classes = 2
epochs = 6

for i in range(2):
    print(str(i) + " EJECUCION DEL PROGRAMA \n")
    
    (x_train, y_train), (x_test, y_test) = load_data(os.getcwd() + "/cats-dogs", img_rows, img_cols)
    
    # Observa si lo que estamos usando como backend es tensorflow o theano y arregla los datos segun el mismo
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else: # No olvidarse de poner tambien si tiene 3 colores o si es solo 1
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)
            
            
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
        
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    
    score = model.evaluate(x_test, y_test, verbose=0)
    
    
    x=0
    file_name = ''
    while True:
        file_name = 'model_' + str(x) + '.h5'
        exists = os.path.isfile( os.getcwd() + '/' + file_name )
        if not exists:
            model.save(file_name)
            break
        else:
            x+=1
            
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    

    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Number of epochs')
    plt.ylabel('categorical_crossentropy')
    plt.grid(True)
    plt.legend()
    
    figure_name = 'figure_of_' + file_name
    plt.savefig(figure_name[:-3]) # El -3 le saca el ".h5" del nombre del modelo

    
