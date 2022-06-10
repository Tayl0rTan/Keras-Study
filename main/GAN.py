import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pylab
from pandas import DataFrame, Series
from keras import models, layers, optimizers, losses, metrics
from keras.utils.np_utils import to_categorical

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#先搞生成器

import keras

latent_dim = 32
height = 32
width = 32
channels = 32

generator_input = keras.Input(shape=(latent_dim,))
x = layers.Dense(128*16*16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16,16,128))(x)

x = layers.Conv2D(256, 5,padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

x=layers.Conv2D(channels,7,activation='tanh',padding='same')(x)
generator = keras.Model(generator_input,x)
generator.summary()

#再搞判别器
discrimination_input=layers.Input(shape=(height,width,channels))

x=layers.Conv2D(128,3)(discrimination_input)
x=layers.LeakyReLU()(x)

x=layers.Conv2D(128,4,strides=2)(x)
x=layers.LeakyReLU()(x)

x=layers.Conv2D(128,4,strides=2)(x)
x=layers.LeakyReLU()(x)

x=layers.Conv2D(128,4,strides=2)(x)
x=layers.LeakyReLU()(x)

x=layers.Flatten()(x)
x=layers.Dropout(0.4)(x)
x=layers.Dense(1,activation='sigmoid')(x)
discriminator=keras.models.Model(discrimination_input,x)
discriminator.summary()

discriminator_optimizer = optimizers.RMSprop(
    lr=0.0008,
    clipvalue=1.0,
    decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
discriminator.trainable=False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)
gan_optimizer = keras.optimizers.RMSprop(
    lr=0.0004,
    clipvalue=1.0,
    decay=1e-8
)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

import os
from  keras.preprocessing import image

(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
x_train = x_train[y_train.flatten() == 6]
x_train = x_train.reshape(
    (x_train.shape[0],) +
    (height, width, channels)).astype('float32') / 255.

iterations = 10000
batch_size = 20
save_dir = '../data/gan_output'
start = 0



#训练DCGAN
"""
每轮都进行以下操作：
    (1) 从潜在空间中抽取随机的点（随机噪声）。
    (2) 利用这个随机噪声用 generator 生成图像。
    (3) 将生成图像与真实图像混合。
    (4) 使用这些混合后的图像以及相应的标签（真实图像为“真”，生成图像为“假”）来训练 discriminator。
    (5) 在潜在空间中随机抽取新的点。
    (6) 使用这些随机向量以及全部是“真实图像”的标签来训练gan。这会更新生成器的权重（只更新生成器的权重，因为判别器在 gan中被冻结），其更新方向是使得判别器能够将生成图像预测为“真实图像”。这个过程是训练生成器去欺骗判别器。
"""
for step in range(iterations):
    random_latent_vectors = np.random.normal(size=(batch_size,latent_dim))
    generated_images=generator.predict(random_latent_vectors)
    stop = start+batch_size
    real_images = x_train[start:stop]

    combined_images=np.concatenate([generated_images,real_images])

    labels = np.concatenate([np.ones((batch_size,1)),np.zeros((batch_size,1))])
    labels += 0.05 * np.random.random(labels.shape)

    d_loss = discriminator.train_on_batch(combined_images, labels)

    random_latent_vectors = np.random.normal(size=(batch_size,latent_dim))
    misleading_targets = np.zeros((batch_size,1))
    a_loss=gan.train_on_batch(
        random_latent_vectors,
        misleading_targets
    )
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
    if step % 100 == 0:
        gan.save_weights('gan.h5')
        print('discriminator loss:', d_loss)
        print('adversarial loss',a_loss)
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'))
        img=image.array_to_img(real_images[0]*255.,scale=False)
        img.save(os.path.join(save_dir,'real_frog'+str(step)+'.png'))

