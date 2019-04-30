"""
8 ENTRADAS Y 5 NEURONA
score:
    ['loss', 'binary_predict']
    [0.01756862576439744, 0.05000000074505806] 
"""


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Setear numero de neuronas
n_neurons = 5
# Setear numero de entradas (solo se puede 2, 5 y 8 segun los txt files)
n_entradas = 8

import pandas as pd
data = pd.read_csv(str(n_entradas)+'.txt', header = None, index_col = False)

x_train=data.values #Convierte lo leido en arrays
y_train=[]
for x in range(len(x_train)):
    y_train=np.append(y_train, np.array(np.prod(x_train[x])))
y_train = np.reshape(y_train, (len(x_train),1))

model = Sequential()
model.add(Dense(n_neurons, input_dim=n_entradas, activation='tanh'))
model.add(Dense(1,activation='tanh'))

import tensorflow as tf
# Esta funcion te retorna la cantidad de elementos que predijo mal (abajo del 0.6)
# Si lo dividimos por el numero de datos de entrenamiento tenemos el porcentaje
# El numero de datos de entrenamiento es len(x_train)
def binary_predict(y_true, y_pred):
    return (tf.reduce_sum(tf.round(tf.abs(y_true-y_pred))))/n_neurons

model.compile(loss='mean_squared_error',
          optimizer=Adam(lr=0.01,),
          metrics=[binary_predict])
model.fit(x_train, y_train, batch_size=16,
      epochs=3000)

score = model.evaluate(x_train, y_train) # los x_test y_test son iguales a los x_train y_train
print('The score of the dimension', n_neurons, 'of', n_entradas, 'entrys is: ', score, '\n')
