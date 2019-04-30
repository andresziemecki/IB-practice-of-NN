""" Andres """

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

x_train = np.array([[0,0],[0,1],[1,0],[1,1]])
y_train = np.array([[0],[1],[1],[0]])
x_test = np.array([[0,0],[0,1],[1,0],[1,1]])
y_test = np.array([[0],[1],[1],[0]])

model = Sequential()
model.add(Dense(2, input_dim=2, activation='tanh'))
model.add(Dropout(0))
model.add(Dense(1, activation='tanh'))

model.compile(loss='mean_squared_error',
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
              epochs=1000,
              batch_size=1)
score = model.evaluate(x_test, y_test, batch_size=1)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
 
