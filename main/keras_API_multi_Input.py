from keras.models import Model
from keras import layers
from keras import Input
from keras import utils

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

text_input = Input(shape=(None,),dtype='int32',name='text')
embedd_text = layers.Embedding(text_vocabulary_size,64)(text_input)
encode_text = layers.LSTM(32)(embedd_text)

question_input = Input(shape=(None,),dtype='int32',name='question')
embedd_question = layers.Embedding(question_vocabulary_size,32)(question_input)
encode_question = layers.LSTM(16)(embedd_question)

concat_tensor = layers.concatenate([encode_text,encode_question],axis=-1)
answer = layers.Dense(answer_vocabulary_size,activation='softmax')(concat_tensor)


model = Model([text_input,question_input],answer)
model.summary()
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])

import numpy as np
num_samples = 1000
max_len = 100

text = np.random.randint(1,text_vocabulary_size,size=(num_samples,max_len))
question = np.random.randint(1,question_vocabulary_size,size=(num_samples,max_len))
answers = np.random.randint(answer_vocabulary_size,size=(num_samples))
answers = utils.to_categorical(answers,answer_vocabulary_size)

model.fit({'text':text,'question':question},answers,epochs=10,batch_size=128)
