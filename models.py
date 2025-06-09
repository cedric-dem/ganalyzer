from config import *

from keras import layers
import tensorflow as tf

if model_name == "test_0":

    def get_discriminator():
        return tf.keras.Sequential(
            [
                layers.Input(shape=(64, 64, 3)),
                layers.Conv2D(64, kernel_size=5, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Conv2D(128, kernel_size=5, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Conv2D(256, kernel_size=5, strides=2, padding="same"),  # lowering size
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Conv2D(512, kernel_size=3, strides=1, padding="same"),  # additional layer
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Flatten(),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

    def get_generator():
        return tf.keras.Sequential(
            [
                layers.Input(shape=(latent_dimension_generator,)),
                # Project and reshape initial (par ex. 8x8x256)
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
                # Output Layer  => (64, 64, 3)
                layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding="same", activation="tanh"),
            ]
        )

elif model_name == "test_0B":

    def get_discriminator():
        FC_size = 128
        return tf.keras.Sequential(
            [
                layers.Input(shape=(64, 64, 3)),
                layers.Conv2D(64, kernel_size=5, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Conv2D(128, kernel_size=5, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Conv2D(256, kernel_size=5, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Conv2D(512, kernel_size=3, strides=1, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Flatten(),
                layers.Dense(FC_size),  # Added fully connected hidden layer
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

    def get_generator():
        FC_size = 512
        return tf.keras.Sequential(
            [
                layers.Input(shape=(latent_dimension_generator,)),
                layers.Dense(FC_size),
                layers.LeakyReLU(),
                layers.Dense(8 * 8 * 256, use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Reshape((8, 8, 256)),
                layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding="same", activation="tanh"),
            ]
        )

elif model_name == "test_0B2":

    def get_discriminator():
        FC_size = 256
        return tf.keras.Sequential(
            [
                layers.Input(shape=(64, 64, 3)),
                layers.Conv2D(64, kernel_size=5, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Conv2D(128, kernel_size=5, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Conv2D(256, kernel_size=5, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Conv2D(512, kernel_size=3, strides=1, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Flatten(),
                layers.Dense(FC_size),  # Added fully connected hidden layer
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

    def get_generator():
        FC_size = 1024
        return tf.keras.Sequential(
            [
                layers.Input(shape=(latent_dimension_generator,)),
                layers.Dense(FC_size),
                layers.LeakyReLU(),
                layers.Dense(8 * 8 * 256, use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Reshape((8, 8, 256)),
                layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding="same", activation="tanh"),
            ]
        )


elif model_name == "test_0B2B":

    def get_discriminator():
        FC_size1 = 256
        FC_size2 = 128
        return tf.keras.Sequential(
            [
                layers.Input(shape=(64, 64, 3)),
                layers.Conv2D(64, kernel_size=5, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Conv2D(128, kernel_size=5, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Conv2D(256, kernel_size=5, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Conv2D(512, kernel_size=3, strides=1, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Flatten(),
                layers.Dense(FC_size1),
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Dense(FC_size2),
                layers.LeakyReLU(alpha=0.2),
                layers.Dropout(0.3),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

    def get_generator():
        FC_size1 = 1024
        FC_size2 = 2048
        return tf.keras.Sequential(
            [
                layers.Input(shape=(latent_dimension_generator,)),
                layers.Dense(FC_size1),
                layers.LeakyReLU(),
                layers.Dense(FC_size2),
                layers.LeakyReLU(),
                layers.Dense(8 * 8 * 256, use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Reshape((8, 8, 256)),
                layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding="same", activation="tanh"),
            ]
        )

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
        nc = 3

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
