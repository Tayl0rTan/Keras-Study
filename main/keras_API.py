from keras.models import Model
from keras import layers
from keras import Input

input_tesor = Input(shape=(64,))
x = layers.Dense(32,activation='relu')(input_tesor)
x = layers.Dense(32,activation='relu')(x)
output_tensor = layers.Dense(10,activation='softmax')(x)
model = Model(input_tesor,output_tensor)
model.summary()
model.compile(optimizer='rmsprop',loss='categorical_crossentropy')

import numpy as np
x_train = np.random.random((1000,64))
y_train = np.random.random((1000,10))
model.fit(x_train,y_train,epochs=10,batch_size=64)
score = model.evaluate(x_train,y_train)
print(score)