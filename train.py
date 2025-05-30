from config import *

import os
import re
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
