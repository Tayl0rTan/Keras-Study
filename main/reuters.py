from keras.datasets import reuters
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers


def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.0
    return results


def to_one_hot(labels, dimension = 46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i,label] = 1.
    return results


(train_data,train_labels), (test_data,test_labels) = reuters.load_data(num_words=10000)
print(len(train_data),len(train_labels))
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
ont_hot_train_labels = to_one_hot(train_labels)
ont_hot_test_labels = to_one_hot(test_labels)

# ont_hot_train_labels = to_categorical(train_labels)
# ont_hot_test_labels = to_categorical(test_labels)



model = models.Sequential()

model.add(layers.Dense(64, activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = ont_hot_train_labels[:1000]
partial_y_train = ont_hot_train_labels[1000:]

history = model.fit(partial_x_train,partial_y_train,epochs=9,batch_size=512,validation_data=(x_val,y_val))


history_dic = history.history
loss_values = history_dic['loss']
val_loss_values = history_dic['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title('training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()