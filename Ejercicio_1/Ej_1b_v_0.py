import numpy as np
from keras.layers import Input, Dense, Concatenate
from keras.models import Model
from keras.optimizers import Adam

x_train = np.array([[0,0],[0,1],[1,0],[1,1]])   
y_train = np.array([[0],[1],[1],[0]])
x_test = np.array([[0,0],[0,1],[1,0],[1,1]])
y_test = np.array([[0],[1],[1],[0]])

inputs = Input(shape=(2,))

layer_1 = Dense(1, activation='tanh')(inputs)
aux = Concatenate()([inputs,layer_1])
layer_2 = Dense(1, activation='tanh')(aux)

model = Model(inputs=inputs, outputs=layer_2)
model.compile(optimizer=Adam(),
              loss='mean_squared_error',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
              epochs=4000,
              batch_size=1,
              validation_data=(x_test,y_test))
score = model.evaluate(x_test, y_test, batch_size=1)


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model).create(prog='dot', format='svg'))

import matplotlib.pyplot as plt

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Number of epochs')
plt.ylabel('Mean squared error')
plt.grid(True)
plt.legend()