from keras import models
from keras import layers
from keras import optimizers

if __name__ == '__main__':
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(784,)))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='me', metrics=['accuracy'])
    model.fit(input_tensor, target_tensor, batch_size=128, epochs=10)