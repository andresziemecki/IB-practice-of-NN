# Trabajo Práctico - Aprendizaje Supervisado
# Ejercicio 1, incisco a) version 0 asd

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Oct 14 15:37:43 2018

@author: andres
"""
"""
# Reseteo lo que ya hab�a anteriormenteasd
from IPython import get_ipython
get_ipython().magic('reset -sf')
"""
from keras.models import Sequential
from keras.layers import Dense, Activation

#Construyo el modelo
model = Sequential()
model.add(Dense(2, input_dim=2))
model.add(Activation('linear'))
model.add(Dense(1))
model.add(Activation('linear'))

from keras import optimizers

sgd = optimizers.SGD(lr=0.01, decay=0, momentum=0, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# No logro entender cual es el objetivo de "Metrics" 


import numpy as np

data=np.array([[1,1],[1,0],[0,1],[0,0]])
labels=np.array([[0],[1],[1],[0]])

# Entreno el modelo
history=model.fit(data, labels, epochs = 10)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.xlabel('Number of epochs')
plt.ylabel('Binary Cross entropy')
plt.grid(True)
plt.legend()

x_test=data
y_test=labels

print("\n\n\nMetrics(Test loss & Test Accuracy): ")
metrics = model.evaluate(x_test, y_test)
print(metrics)

from keras.utils import plot_model
plot_model(model, to_file='model.png')