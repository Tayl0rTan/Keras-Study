import os

data_dir = r'/Users/taylor/Documents/python/keras_study/data/jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

import numpy as np

float_data = np.zeros((len(lines), len(header) - 1))
for i,line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i,:] = values

mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

def generator(data, lookback, delay, min_index, max_index,shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        #是否随机打乱取数据
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step,data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

#准备训练生成器、验证生成器和测试生成器
lookback = 1440
step = 6
delay = 144
batch_size = 128
train_gen = generator(float_data,lookback=lookback,delay=delay,min_index=0,
                      max_index=200000,shuffle=True,step=step,batch_size=batch_size)

val_gen = generator(float_data,lookback=lookback,delay=delay,min_index=200001,
                    max_index=300000,step=step,batch_size=batch_size)

test_gen = generator(float_data,lookback=lookback,delay=delay,min_index=300001,
                     max_index=None,step=step,batch_size=batch_size)

val_steps = (300000 - 200001 - lookback) //batch_size
test_steps = (len(float_data) - 300001 - lookback) //batch_size

#对比1：简单的规则方法，当前温度 = 前一天的温度
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples,targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))
celsius_mae = 2897359729905486 * std[1]
print(celsius_mae)

from keras.models import Sequential
from keras.layers import Dense, Bidirectional, GRU
from keras.optimizers import RMSprop

model = Sequential()

#对比2：简单的全连接层 利用Flatten层将输入摊平
# model.add(Flatten(input_shape=(lookback//step, float_data.shape[-1])))
# model.add(Dense(32,activation='relu'))

#对比3：使用GRU层来处理序列数据
model.add(GRU(32,input_shape=(None, float_data.shape[-1])))

#对比4：给GRU层增加drop来减缓过拟合
# model.add(GRU(32,dropout=0.2, recurrent_dropout=0.2,input_shape=(None, float_data.shape[-1])))

#对比5：使用两层GRU层
# model.add(GRU(32,dropout=0.1, recurrent_dropout=0.5,return_sequences=True,input_shape=(None, float_data.shape[-1])))
# model.add(GRU(64,dropout=0.1, recurrent_dropout=0.5,input_shape=(None, float_data.shape[-1])))

#对比6：使用双向GRU
# model.add(Bidirectional(GRU(32),input_shape=(None, float_data.shape[-1])))

model.add(Dense(1))
model.compile(optimizer=RMSprop(),loss=['mae'])
history = model.fit_generator(train_gen,steps_per_epoch=500, epochs=40, validation_data=val_gen, validation_steps=val_steps)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
