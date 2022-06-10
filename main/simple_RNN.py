from keras.datasets import imdb
from keras.preprocessing import sequence

max_feature = 10000
max_len = 500
batch_size = 32

print('Loading data...')

(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_feature)
print(len(input_train), 'dogs_vs_cats sequence')
print(len(input_test), 'test sequence')

print('Pad sequence (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=max_len)
input_test = sequence.pad_sequences(input_test, maxlen=max_len)
print('input_train shape', input_train.shape)
print('input_test shape', input_test.shape)

from keras.layers import Dense, Embedding, SimpleRNN, LSTM
from keras.models import Sequential

model = Sequential()
model.add(Embedding(max_feature, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train,y_train,epochs=10,batch_size=128,validation_data=(input_test,y_test))


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)
plt.plot(epochs, acc, 'bo', label='Training Acc')
plt.plot(epochs, val_acc, 'b', label='Validation Acc')
plt.title('Training and Validation Accuary')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
