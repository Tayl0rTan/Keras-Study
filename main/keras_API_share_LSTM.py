from keras.models import Model
from keras import layers
from keras import Input

lstm = layers.LSTM(32)

left_input = Input(shape=(64,))
left_output = lstm(left_input)

right_input = Input(shape=(64,))
right_output = lstm(right_input)

merged = layers.concatenate([left_output, right_output],axis=-1)
predictions = layers.Dense(1,activation='sigmoid')(merged)

model = Model([left_input,right_input],predictions)
model.fit([left_data, right_data],output_data)
