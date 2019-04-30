#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 00:44:01 2018

@author: andres
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
x_train = np.random.rand(50)
x_test = np.random.rand(50)

m = 2 #Pendiente de la regresion
# Genero los datos de salida
y_train = x_train*m
Gauss_distr = np.random.normal(0.5, 0.4, 50)
y_train+=Gauss_distr
y_test = x_test*m
Gauss_distr_2 = np.random.normal(0.5, 0.4, 50)
y_test+=Gauss_distr_2


model = Sequential()
model.add(Dense(1, activation='linear', input_dim=1))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size = 1, epochs=20, verbose = 1,
                    validation_data=(x_test, y_test))

prediction = model.predict(x_test, batch_size = 1, verbose = 0)

#Regresion lineal
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(x_test,y_test)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(x_test, y_test, 'ro')
plt.plot(x_test, prediction, 'bo')
plt.plot(x_test, x_test*slope + intercept, 'g')
# plt.plot(x_test, x_test*m + 0.5, 'g') Esto seria en vez de un ajuste lineal,
# seria la pendiente que yo aplique a los datos x mas el valor medio de la gaussiana
plt.xlabel('Number of epochs')
plt.ylabel('Mean squared error')
plt.grid(True)
plt.legend()
plt.show()



