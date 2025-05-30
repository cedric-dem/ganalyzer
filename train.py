from config import *

import os
import re
import csv
import time
from tqdm import tqdm

import cv2
import keras
from keras import layers
from keras.preprocessing.image import img_to_array
import numpy as np
import tensorflow as tf

def getGenerator():

    # Size of feature maps in generator
    ngf = 64

    # Number of channels in the training images. For color images this is 3
    nc = 3

    return tf.keras.Sequential([
        # couche 1: (latent_dimension_generator,) -> (ngf*8, 4, 4)
        #layers.Input(shape=(1, 1, nz)),
        layers.Input(shape=(latent_dimension_generator,)),
        layers.Reshape((1, 1, latent_dimension_generator)),
        layers.Conv2DTranspose(ngf * 8, kernel_size=4, strides=1, padding='valid', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        # (ngf*8, 4, 4) -> (ngf*4, 8, 8)
        layers.Conv2DTranspose(ngf * 4, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        # (ngf*4, 8, 8) -> (ngf*2, 16, 16)
        layers.Conv2DTranspose(ngf * 2, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        # (ngf*2, 16, 16) -> (ngf, 32, 32)
        layers.Conv2DTranspose(ngf, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        # (ngf, 32, 32) -> (nc, 64, 64)
        layers.Conv2DTranspose(nc, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.Activation('tanh')
    ])


def getDiscriminator():
    if rgb_images:
        nc = 3
    else:
        nc=1

    # Size of feature maps in discriminator
    ndf = 64

    return tf.keras.Sequential([
        # input is (64, 64, nc)
        layers.Conv2D(ndf, kernel_size=4, strides=2, padding='same', use_bias=False, input_shape=(64, 64, nc)),
        layers.LeakyReLU(alpha=0.2),

        # (32, 32, ndf)
        layers.Conv2D(ndf * 2, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        # (16, 16, ndf*2)
        layers.Conv2D(ndf * 4, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        # (8, 8, ndf*4)
        layers.Conv2D(ndf * 8, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        # (4, 4, ndf*8)
        layers.Conv2D(1, kernel_size=4, strides=1, padding='valid', use_bias=False),
        layers.Activation('sigmoid')
        # Output is (1, 1, 1)
    ])

def train(current_epoch, dataset, generator, discriminator, cross_entropy):

    for epoch in range(current_epoch, 999):
        print("==> current epoch : ",epoch)

        generator.save( model_path+'generator_epoch_' + str(epoch) + ".keras")
        discriminator.save( model_path+ 'discriminator_epoch_' + str(epoch) + ".keras")

        start = time.time()

        total_stats = {
            "median_real": 0,
            "median_fake": 0,
            "mean_real": 0,
            "mean_fake": 0,
            'gen_loss': 0,
            'disc_loss': 0
        }

        for batch in dataset:
            this_stats = train_steps(batch, cross_entropy)

            for key in this_stats:
                total_stats[key] += this_stats[key]

        total_stats['time'] = str(np.round(time.time() - start, 2))

        addStatsToFile(epoch, total_stats)

def addStatsToFile(epoch, newStats):

    exists = os.path.isfile(statistics_file_path)

    with open(statistics_file_path, mode='a', newline='', encoding='utf-8') as statfile:
        writer = csv.writer(statfile)

        if not exists:
            writer.writerow(["epoch_id","median_real","median_fake","mean_real","mean_fake", 'gen_loss','disc_loss', "time"])

        writer.writerow([str(epoch)] + [newStats[key] for key in newStats])

def train_steps(images, cross_entropy):
    # TODO
    time.sleep(0.03)
    return {
        "median_real": 1,
        "median_fake": 3,
        "mean_real": 5,
        "mean_fake": 5,
        'gen_loss': 2,
        'disc_loss': 45
    }

def generator_loss(fake_output, cross_entropy):
    return cross_entropy(tf.ones_like(fake_output),fake_output)

def discriminator_loss(fake_output, real_output, cross_entropy):
    fake_loss = cross_entropy(tf.zeros_like(fake_output),fake_output)
    real_loss = cross_entropy(tf.ones_like(real_output),real_output)
    return fake_loss + real_loss

def getModelQuantity(filename):
    current_i=0
    cont=True

    while cont:
        cont=os.path.isfile(filename+str(current_i)+'.keras')
        current_i+=1

    return current_i-2

def getCurrentEpoch():
    counterGenerator = getModelQuantity(model_path + 'generator_epoch_')
    counterDiscriminator = getModelQuantity(model_path + 'discriminator_epoch_')

    return max(min(counterGenerator, counterDiscriminator),0)

def sortedAlphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
    return sorted(data,key = alphanum_key)

def getDataset():
    _img = []

    files = os.listdir(dataset_path)
    files = sortedAlphanumeric(files)
    for i in tqdm(files):
        if rgb_images:
            img = cv2.cvtColor(cv2.imread(os.path.join(dataset_path, i), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(os.path.join(dataset_path, i), cv2.IMREAD_GRAYSCALE)

        # resizing image
        # img = cv2.resize(img, (SIZE, SIZE))
        img = (img - 127.5) / 127.5
        _img.append(img_to_array(img))
    return _img

def launchTraining():
    currentEpoch = getCurrentEpoch()
    print("==> will start from epoch  : ", currentEpoch)

    _img=getDataset()

    batch_size = 32
    dataset = tf.data.Dataset.from_tensor_slices(np.array(_img)).batch(batch_size)

    if currentEpoch == 0: #if start from scratch
        print('==> Creating models')
        generator = getGenerator()
        discriminator = getDiscriminator()

    else:
        print('==> Loading latest models')

        discriminator = keras.models.load_model( model_path+'discriminator_epoch_' + str(currentEpoch) + ".keras")
        generator = keras.models.load_model( model_path+'generator_epoch_' + str(currentEpoch) + ".keras")

    generator.summary()
    discriminator.summary()

    generator_optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=.0001,
        clipvalue=1.0,
    )

    discriminator_optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=.0001,
        clipvalue=1.0,
    )

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    print('==> Number of batches : ',len(dataset))
    train(currentEpoch, dataset, generator, discriminator, cross_entropy)

launchTraining()