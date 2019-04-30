import numpy as np
from keras.layers import Input, Dense, Concatenate
from keras.models import Model
from keras.optimizers import Adam


y=0.1
x_train = np.array([[y]])
y=4*y*(1-y)
y_train = np.array([[y]])
x_test = np.array([[y]])
y_test = np.array([[4*y*(1-y)]])

for i in range(1,100):
    if (i<= 50):
        x_train = np.append(x_train, [[y]])
        y=4*y*(1-y)
        y_train = np.append(y_train, [[y]])
    if (i>50):
        x_test = np.append(x_test, [[y]])
        y=4*y*(1-y)
        y_test = np.append(y_test, [[y]])
        

inputs = Input(shape=(1,))

layer_1 = Dense(5, activation='tanh')(inputs)
aux = Concatenate()([inputs,layer_1])
layer_2 = Dense(1, activation='linear')(aux)

model = Model(inputs=inputs, outputs=layer_2)
model.compile(optimizer=Adam(lr=0.1),
              loss='mean_squared_error',
              metrics=["mae"])

history = model.fit(x_train, y_train,
              epochs=500,
              validation_data=(x_test,y_test))


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model).create(prog='dot', format='svg'))

prediction = model.predict(x_test, verbose = 0)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(x_test, y_test, 'ro')
plt.plot(x_train, y_train, 'go')
plt.plot(x_test, prediction, 'bo')

# plt.plot(x_test, x_test*m + 0.5, 'g') Esto seria en vez de un ajuste lineal,
# seria la pendiente que yo aplique a los datos x mas el valor medio de la gaussiana
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)


plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Number of epochs')
plt.ylabel('Mean squared error')
plt.grid(True)
plt.legend()
