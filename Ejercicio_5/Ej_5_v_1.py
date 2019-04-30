import numpy as np
import keras
from keras.optimizers import Adam

f = open("pima-indians-diabetes.csv", "r")
x_train = np.empty((0,8), int)
x_test = np.empty((0,8), int)
y_train = np.empty((0,1), int)
y_test = np.empty((0,1), int)
nTrain = 500    #numero de datos para usarse como train (el resto son para test)
                #recordar que el archivo tiene 768 lineas
xDen = np.array([17, 199, 122, 99, 846, 67.1, 2420, 81]);   #normalization
yDen = np.array([1]);                                       #normalization
i = 0
for line in f:
    i = i+1
    lineContent = line.split(",")
    xNewRow = [lineContent[0:8:1]]
    xNewRow = np.array(xNewRow).astype(float) / xDen
    yNewRow = [lineContent[8:9:1]]
    yNewRow = np.array(yNewRow).astype(float) / yDen
    if(i<=nTrain):
        x_train = np.vstack([x_train, xNewRow])
        y_train = np.vstack([y_train, yNewRow])
    else:
        x_test = np.vstack([x_test, xNewRow])
        y_test = np.vstack([y_test, yNewRow])

Input = keras.layers.Input(shape=(8,))
Layer_1 = keras.layers.Dense(8, activation='relu')(Input)
Layer_2 = keras.layers.Dense(8, activation='relu')(Layer_1)
Residual_1 = keras.layers.Add()([Input, Layer_1])
Layer_3 = keras.layers.Dense(8, activation='relu')(Residual_1)
Layer_4 = keras.layers.Dense(8,activation='relu')(Layer_3)
Residual_2 = keras.layers.Add()([Residual_1, Layer_4])
Layer_5 = keras.layers.Dense(8, activation='relu')(Residual_2)
Layer_6 = keras.layers.Dense(8,activation='relu')(Layer_5)
Residual_3 = keras.layers.Add()([Residual_2, Layer_6])
out = keras.layers.Dense(1,activation='sigmoid')(Residual_3)

model = keras.models.Model(inputs=[Input], outputs=out)

model.compile(loss='mean_squared_error',
              optimizer=Adam(),
              metrics=['accuracy'])

#Entrenamos la red
history = model.fit(x_train, y_train,
              epochs=500,
              batch_size=100,
              validation_data=(x_test,y_test))
score = model.evaluate(x_test, y_test)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Number of epochs')
plt.ylabel('Mean squared error')
plt.grid(True)
plt.legend()

#Cierro el archivo
f.close()