#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 13:53:13 2019

@author: andres
"""

import os
import imageio
import numpy as np
import PIL
from PIL import ImageOps
import keras

def load_data(path, image_height = 128, image_width = 128, shuffle=True):

    # Arma una lista fnames con todos los nombres de los archivos que hay en el directorio path
    fnames = [f for f in os.listdir(path) if f.startswith('c') or
            f.startswith('d') and f.endswith('.jpg')]
    
    # Mezcla los nombres
    if shuffle:
        np.random.shuffle(fnames)

    # labels = 0 for cats and 1 for dogs
    labels = [0 if f.split('.')[0] == 'cat' else 1 for f in fnames]
    
    # Lista donde se colocaran todas las imagenes
    imgs = []

    # Añade cada imagen dentro de la lista imgs
    for f in fnames:
        # Lee una imagen y la coloca en I
        I = imageio.imread(os.path.join(path, f))#.astype('float32')/255
        # En imageOps.fit necesito una imagen tipo PIL, paso de tipo imageio a array y luego a PIL
        I = np.asarray(I)
        I = PIL.Image.fromarray(I)
        # ImageOps.fit le hace un resize
        I = ImageOps.fit(I, [image_width,image_height])
        # Vuelve a convertirlo en un Numpy array
        I = np.array(I)
        # Lo añade a la lista
        imgs.append(I)

    # Transformo la lista que tenia en una lista numpy array
    labels = np.array(labels)

    # Separo lo que es train de lo que es test en un 70% y 30%
    imgs_train, imgs_test = np.split(imgs, [round(0.7*len(fnames))], axis=0)
    labels_train, labels_test = np.split(labels, [round(0.7*len(fnames))], axis=0)
    
    # Normalizo las imagenes y las transformo a tipo float porque eran int
    imgs_train = imgs_train.astype('float32')/255
    imgs_test = imgs_test.astype('float32')/255
    
    return (imgs_train, labels_train), (imgs_test, labels_test)



def red_miocardica(input_shape):
    
    Input = keras.layers.Input(shape=input_shape)

    Layer_1 = keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same",
                 activation='relu')(Input)
    
    norm_1 = keras.layers.BatchNormalization()(Layer_1)


    Layer_2 = keras.layers.Conv2D(64, kernel_size=(3, 3),padding="same",
                 activation='relu')(norm_1)

    norm_2 = keras.layers.BatchNormalization()(Layer_2)

    Proyection = keras.layers.Conv2D(64, kernel_size=(1,1), activation='linear')(Input)

    Residual_1 = keras.layers.Add()([Proyection, norm_2])

    # **************SEGUNDA FILA DE CAPAS***************

    MaxPool_1 = keras.layers.MaxPool2D(pool_size=(2,2))(Residual_1)

    Layer_3 = keras.layers.Conv2D(128, kernel_size=(3, 3),padding="same",
                 activation='relu')(MaxPool_1)
    
    norm_3 = keras.layers.BatchNormalization()(Layer_3)
        
    Layer_4 = keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same",
                                  activation='relu')(norm_3)
    
    norm_4 = keras.layers.BatchNormalization()(Layer_4)
    
    Proyection_2 = keras.layers.Conv2D(128, kernel_size=(1,1), activation='linear')(MaxPool_1)
    
    Residual_2 = keras.layers.Add()([Proyection_2, norm_4])
    
    
    
    #************ TERCER FILA DE CAPAS ***********
    
    MaxPool_2 = keras.layers.MaxPool2D(pool_size=(2,2))(Residual_2)
    
    Layer_5 = keras.layers.Conv2D(256, kernel_size=(3, 3),padding="same",
                                  activation='relu')(MaxPool_2)
    
    norm_5 = keras.layers.BatchNormalization()(Layer_5)
    
    Layer_6 = keras.layers.Conv2D(256, kernel_size=(3, 3), padding="same",
                                  activation='relu')(norm_5)
    
    norm_6 = keras.layers.BatchNormalization()(Layer_6)
    
    Proyection_3 = keras.layers.Conv2D(256, kernel_size=(1,1), activation='linear')(MaxPool_2)
    
    Residual_3 = keras.layers.Add()([Proyection_3, norm_6])
    
    #*********** CUARTA FILA DE CAPAS **********
    
    MaxPool_3 = keras.layers.MaxPool2D(pool_size=(2,2))(Residual_3)
    
    Layer_7 = keras.layers.Conv2D(512, kernel_size=(3, 3),padding="same",
                                  activation='relu')(MaxPool_3)
    
    norm_7 = keras.layers.BatchNormalization()(Layer_7)
    
    Layer_8 = keras.layers.Conv2D(512, kernel_size=(3, 3), padding="same",
                                  activation='relu')(norm_7)
    
    norm_8 = keras.layers.BatchNormalization()(Layer_8)
    
    Proyection_4 = keras.layers.Conv2D(512, kernel_size=(1,1), activation='linear')(MaxPool_3)
    
    Residual_4 = keras.layers.Add()([Proyection_4, norm_8])
    
    # *********** QUINTA FILA DE CAPAS ***********
    
    MaxPool_4 = keras.layers.MaxPool2D(pool_size=(2,2))(Residual_4)
    
    Layer_9 = keras.layers.Conv2D(1024, kernel_size=(3, 3),padding="same",
                                  activation='relu')(MaxPool_4)
    
    norm_9 = keras.layers.BatchNormalization()(Layer_9)
    
    Layer_10 = keras.layers.Conv2D(1024, kernel_size=(3, 3), padding="same",
                                   activation='relu')(norm_9)
    
    norm_10 = keras.layers.BatchNormalization()(Layer_10)
    
    Proyection_5 = keras.layers.Conv2D(1024, kernel_size=(1,1), activation='linear')(MaxPool_4)
    
    Residual_5 = keras.layers.Add()([Proyection_5, norm_10])
    
    # *************PRIMER FILA DE CAPAS EN SUBIDA*************
    
    
    Up_Pooling_1 = keras.layers.UpSampling2D(size=(2,2))(Residual_5)
    
    Proyeccion_Up_P_1 = keras.layers.Conv2D(512, kernel_size=(1,1), activation='linear')(Up_Pooling_1)
    
    Concatenacion_1 = keras.layers.Concatenate()([Proyeccion_Up_P_1, Residual_4])
    
    Layer_11 = keras.layers.Conv2D(512, kernel_size=(3, 3), padding="same",
                                   activation='relu')(Concatenacion_1)
    
    norm_11 = keras.layers.BatchNormalization()(Layer_11)
    
    Layer_12 = keras.layers.Conv2D(512, kernel_size=(3, 3), padding="same",
                                   activation='relu')(norm_11)
    
    norm_12 = keras.layers.BatchNormalization()(Layer_12)
    
    Proyection_6 = keras.layers.Conv2D(512, kernel_size=(1,1), activation='linear')(Concatenacion_1)
    
    Residual_6 = keras.layers.Add()([Proyection_6, norm_12])
    
    Residual_7 = keras.layers.Add()([Proyeccion_Up_P_1, Residual_6])
    
    
    # **************** SEGUNDA CAPA DE SUBIDA*****************
    
    Up_Pooling_2 = keras.layers.UpSampling2D(size=(2,2))(Residual_7)
    
    Proyeccion_Up_P_2 = keras.layers.Conv2D(256, kernel_size=(1,1), activation='linear')(Up_Pooling_2)
    
    
    Concatenacion_2 = keras.layers.Concatenate()([Proyeccion_Up_P_2, Residual_3])
    
    
    Layer_13 = keras.layers.Conv2D(256, kernel_size=(3, 3),padding="same",
                                   activation='relu')(Concatenacion_2)
    
    norm_13 = keras.layers.BatchNormalization()(Layer_13)
    
    Layer_14 = keras.layers.Conv2D(256, kernel_size=(3, 3), padding="same",
                                   activation='relu')(norm_13)
    
    norm_14 = keras.layers.BatchNormalization()(Layer_14)
    
    Proyection_7 = keras.layers.Conv2D(256, kernel_size=(1,1), activation='linear')(Concatenacion_2)
    
    Residual_8 = keras.layers.Add()([Proyection_7, norm_14])
    
    Residual_9 = keras.layers.Add()([Proyeccion_Up_P_2, Residual_8])
    
    
    # **************** TERCER CAPA DE SUBIDA*****************
    
    Up_Pooling_3 = keras.layers.UpSampling2D(size=(2,2))(Residual_9)
    
    Proyeccion_Up_P_3 = keras.layers.Conv2D(128, kernel_size=(1,1), activation='linear')(Up_Pooling_3)
    
    Concatenacion_3 = keras.layers.Concatenate()([Proyeccion_Up_P_3, Residual_2])
    
    Layer_15 = keras.layers.Conv2D(128, kernel_size=(3, 3),padding="same",
                                   activation='relu')(Concatenacion_3)
    
    norm_15 = keras.layers.BatchNormalization()(Layer_15)
    
    Layer_16 = keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same",
                                   activation='relu')(norm_15)
    
    norm_16 = keras.layers.BatchNormalization()(Layer_16)
    
    Proyection_8 = keras.layers.Conv2D(128, kernel_size=(1,1), activation='linear')(Concatenacion_3)
    
    Residual_10 = keras.layers.Add()([Proyection_8, norm_16])
    
    Residual_11 = keras.layers.Add()([Proyeccion_Up_P_3, Residual_10])
    
    
    # **************** CUARTA CAPA DE SUBIDA*****************
    
    Up_Pooling_4 = keras.layers.UpSampling2D(size=(2,2))(Residual_11)
    
    Proyeccion_Up_P_4 = keras.layers.Conv2D(64, kernel_size=(1,1), activation='linear')(Up_Pooling_4)
    
    Concatenacion_4 = keras.layers.Concatenate()([Proyeccion_Up_P_4, Residual_1])
    
    Layer_17 = keras.layers.Conv2D(64, kernel_size=(3, 3),padding="same",
                                   activation='relu')(Concatenacion_4)
    
    norm_17 = keras.layers.BatchNormalization()(Layer_17)
    
    Layer_18 = keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same",
                                   activation='relu')(norm_17)
    
    norm_18 = keras.layers.BatchNormalization()(Layer_18)
    
    Proyection_9 = keras.layers.Conv2D(64, kernel_size=(1,1),
                                       activation='linear')(Concatenacion_4)
    
    Residual_12 = keras.layers.Add()([Proyection_9, norm_18])
    
    Residual_13 = keras.layers.Add()([Proyeccion_Up_P_4, Residual_12])
    
    # ******************SALIDA*************************
    
    casi_out = keras.layers.Flatten()(Residual_13)
    
    out = keras.layers.Dense(2,activation='softmax')(casi_out)
    
    """
    out = keras.layers.Conv2D(2, kernel_size=(1, 1),
                              activation='relu')(Residual_13)
    """
    
    model = keras.models.Model(inputs=[Input], outputs=out)
    
    
    return model