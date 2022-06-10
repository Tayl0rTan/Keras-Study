import numpy as np

samples = ['the cat sat on the mat', 'the dog ate my homework']

'''
token级别onehot编码
'''
token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1

max_length = 10  #限制输入的每个序列的最大长度，例子中最长就是5，看不出效果，如果是更长的序列就可以看出

#初始化一个三维张量，第一维是序列的数量，第二维是序列的长度，也就是token数量，第三维是每个token对应向量化长度
results = np.zeros(shape=(len(samples),max_length,max(token_index.values())+1))


for i, sample in enumerate(samples):
    for j,word in list(enumerate(sample.split()))[:max_length]: #max_length 用来限制序列中token的数量
        index = token_index.get(word)
        results[i,j,index] = 1.

print(results)

'''
单个字符级别onehot编码
'''
import string

characters = string.printable #所有可以打印的ascii码字符
token_index = dict(zip(characters,range(1, len(characters) + 1)))

max_length = 50
results = np.zeros(shape=(len(samples),max_length,max(token_index.values())+1))

for i, sample in enumerate(samples):
    for j,character in enumerate(sample[:max_length]): #max_length 用来限制序列中token的数量
        index = token_index.get(character)
        results[i,j,index] = 1.

print(results)

'''
利用keras自带的Tokenizer进行onehot编码
不过我发现这个是文档级别编码 不是单词级别编码
单次级别编码还需要自己实现
'''

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
one_hot_results = tokenizer.texts_to_matrix(samples,mode='binary')
print(sequences)

'''
onehot的所谓散列技巧
'''

dimensionality = 1000
max_length = 10

results  = np.zeros((len(samples), max_length, dimensionality))
for i,sample in enumerate(samples):
    for j,word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word))%dimensionality
        results[i,j,index] = 1.
print(results)