from keras.models import Model
from keras import layers
from keras import Input
from keras import utils

vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None,),dtype='int32',name='posts')
embedd_posts = layers.Embedding(vocabulary_size,256)(posts_input)
x = layers.Conv1D(128,5,activation='relu')(embedd_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dense(128,activation='relu')(x)

pre_age = layers.Dense(1,name='age')(x)
pre_income = layers.Dense(num_income_groups,activation='softmax',name='income')(x)
pre_gender = layers.Dense(1,activation='sigmoid',name='gender')(x)

model = Model(posts_input,[pre_age,pre_income,pre_gender])
model.compile(optimizer='rmsprop',loss = ['mse','categorical_crossentropy','binary_crossentropy'],loss_weights=[0.25,1.,10.])
model.fit(posts,[y_age,y_income,y_gender],batch_size=64,epochs=100)


question_input = Input(shape=(None,),dtype='int32',name='question')
embedd_question = layers.Embedding(question_vocabulary_size,32)(question_input)
encode_question = layers.LSTM(16)(embedd_question)

concat_tensor = layers.concatenate([embedd_text,embedd_question],axis=-1)
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

model.fit([text,question],answers,epochs=10,batch_size=128)
