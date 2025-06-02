from config import *

from keras import layers
import tensorflow as tf

if model_name == "test_0":

    def get_discriminator():
        model = tf.keras.Sequential(
            [
                layers.Input(shape=(64, 64, 3)),
                layers.Conv2D(64, kernel_size=5, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Conv2D(128, kernel_size=5, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Conv2D(256, kernel_size=5, strides=2, padding="same"),  # descente en taille
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Conv2D(512, kernel_size=3, strides=1, padding="same"),  # couche supplémentaire pour complexifier
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Flatten(),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        return model

    def get_generator():
        model = tf.keras.Sequential(
            [
                layers.Input(shape=(latent_dimension_generator,)),
                # Projection et reshape initial (par ex. 8x8x256)
                layers.Dense(8 * 8 * 256, use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Reshape((8, 8, 256)),  # => (8, 8, 256)
                # Upsampling 1 => (16, 16, 128)
                layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                # Upsampling 2 => (32, 32, 64)
                layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                # Upsampling 3 => (64, 64, 32)
                layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                # Couche de sortie => (64, 64, 3)
                layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding="same", activation="tanh"),
            ]
        )
        return model


elif model_name == "test_1":

    def get_generator():

        # Size of feature maps in generator
        ngf = 64

        # Number of channels in the training images. For color images this is 3
        nc = 3

        return tf.keras.Sequential(
            [
                # layer 1: (latent_dimension_generator,) -> (ngf*8, 4, 4)
                # layers.Input(shape=(1, 1, nz)),
                layers.Input(shape=(latent_dimension_generator,)),
                layers.Reshape((1, 1, latent_dimension_generator)),
                layers.Conv2DTranspose(ngf * 8, kernel_size=4, strides=1, padding="valid", use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
                # (ngf*8, 4, 4) -> (ngf*4, 8, 8)
                layers.Conv2DTranspose(ngf * 4, kernel_size=4, strides=2, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
                # (ngf*4, 8, 8) -> (ngf*2, 16, 16)
                layers.Conv2DTranspose(ngf * 2, kernel_size=4, strides=2, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
                # (ngf*2, 16, 16) -> (ngf, 32, 32)
                layers.Conv2DTranspose(ngf, kernel_size=4, strides=2, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
                # (ngf, 32, 32) -> (nc, 64, 64)
                layers.Conv2DTranspose(nc, kernel_size=4, strides=2, padding="same", use_bias=False),
                layers.Activation("tanh"),
            ]
        )

    def get_discriminator():
        if rgb_images:
            nc = 3
        else:
            nc = 1

        # Size of feature maps in discriminator
        ndf = 64

        return tf.keras.Sequential(
            [
                # input is (64, 64, nc)
                layers.Conv2D(ndf, kernel_size=4, strides=2, padding="same", use_bias=False, input_shape=(64, 64, nc)),
                layers.LeakyReLU(alpha=0.2),
                # (32, 32, ndf)
                layers.Conv2D(ndf * 2, kernel_size=4, strides=2, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.2),
                # (16, 16, ndf*2)
                layers.Conv2D(ndf * 4, kernel_size=4, strides=2, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.2),
                # (8, 8, ndf*4)
                layers.Conv2D(ndf * 8, kernel_size=4, strides=2, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.2),
                # (4, 4, ndf*8)
                layers.Conv2D(1, kernel_size=4, strides=1, padding="valid", use_bias=False),
                layers.Activation("sigmoid"),
                # Output is (1, 1, 1)
            ]
        )

else:
    raise Exception("model not found")
