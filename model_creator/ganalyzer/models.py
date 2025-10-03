from config import *

from keras import layers
import tensorflow as tf
import math


MODEL_CONFIGS_64 = {
    "model_0_small": {
        "gen_base": [256, 128, 64, 32, 16, 8],
        "gen_min": 8,
        "disc_seq": [32, 64, 128, 256, 256, 256],
        "disc_fc": lambda image_size: [],
        "gen_ch0": 256,
        "gen_pre_dense": [],
        "extra_conv": False,
    },
    "model_1_medium": {
        "gen_base": [512, 256, 128, 64, 32, 16],
        "gen_min": 16,
        "disc_seq": [64, 128, 256, 512, 512, 512],
        "disc_fc": lambda image_size: [128],
        "gen_ch0": 512,
        "gen_pre_dense": [],
        "extra_conv": True,
    },
    "model_2_large": {
        "gen_base": [512, 256, 128, 64, 32, 16],
        "gen_min": 16,
        "disc_seq": [64, 128, 256, 512, 512, 512],
        "disc_fc": lambda image_size: [image_size * 4, image_size * 2],
        "gen_ch0": 512,
        "gen_pre_dense": [1024],
        "extra_conv": True,
    },
    "model_3_extra_large": {
        "gen_base": [1024, 512, 256, 128, 64, 32],
        "gen_min": 32,
        "disc_seq": [128, 256, 512, 1024, 1024, 1024],
        "disc_fc": lambda image_size: [image_size * 8, image_size * 4, image_size * 2],
        "gen_ch0": 1024,
        "gen_pre_dense": [2048, 1024],
        "extra_conv": True,
    },
}

MODEL_CONFIGS_128 = {
    "model_0_small": {
        "gen_base": [512, 256, 128, 64, 32, 16, 8],
        "gen_min": 8,
        "disc_seq": [64, 128, 256, 512, 512, 512, 512],
        "disc_fc": lambda image_size: [],
        "gen_ch0": 512,
        "gen_pre_dense": [],
        "extra_conv": False,
    },
    "model_1_medium": {
        "gen_base": [1024, 512, 256, 128, 64, 32, 16],
        "gen_min": 16,
        "disc_seq": [128, 256, 512, 1024, 1024, 1024, 1024],
        "disc_fc": lambda image_size: [256],
        "gen_ch0": 1024,
        "gen_pre_dense": [],
        "extra_conv": True,
    },
    "model_2_large": {
        "gen_base": [1024, 512, 256, 128, 64, 32, 16],
        "gen_min": 16,
        "disc_seq": [128, 256, 512, 1024, 1024, 1024, 1024],
        "disc_fc": lambda image_size: [image_size * 4, image_size * 2],
        "gen_ch0": 1024,
        "gen_pre_dense": [2048],
        "extra_conv": True,
    },
    "model_3_extra_large": {
        "gen_base": [2048, 1024, 512, 256, 128, 64, 32],
        "gen_min": 32,
        "disc_seq": [256, 512, 1024, 2048, 2048, 2048, 2048],
        "disc_fc": lambda image_size: [image_size * 8, image_size * 4, image_size * 2],
        "gen_ch0": 2048,
        "gen_pre_dense": [4096, 2048],
        "extra_conv": True,
    },
}

dataset_dimension = 128

if (dataset_dimension == 64):
    MODEL_CONFIGS = MODEL_CONFIGS_64
elif (dataset_dimension == 128):
    MODEL_CONFIGS = MODEL_CONFIGS_128


def _num_upsamples_to_reach(img_size, base=4):
    if img_size < base:
        raise ValueError("img_size should be >= 4")
    cur = base
    ups = 0
    while cur < img_size:
        cur *= 2
        ups += 1
    return ups, cur


def _filters_for_gen(step_idx, base, min_filter):
    return base[step_idx] if step_idx < len(base) else max(min_filter, base[-1] // 2)


def _filters_for_disc(step_idx, seq):
    return seq[step_idx] if step_idx < len(seq) else seq[-1]


def get_discriminator():
    cfg = MODEL_CONFIGS[model_name]
    image_size = int(dataset_dimension)

    assert image_size == int(image_size)
    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = inputs

    cur = image_size
    step = 0
    while cur > 4:
        x = layers.Conv2D(
            _filters_for_disc(step, cfg["disc_seq"]),
            kernel_size=5,
            strides=2,
            padding="same",
        )(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
        cur = math.ceil(cur / 2)
        step += 1

    x = layers.GlobalAveragePooling2D()(x)
    for size in cfg["disc_fc"](image_size):
        x = layers.Dense(size)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs, name=f"Discriminator_{image_size}")


def get_generator():
    cfg = MODEL_CONFIGS[model_name]
    image_size = int(dataset_dimension)
    latent_dim = latent_dimension_generator
    base_spatial = 4

    assert image_size == int(image_size)
    ups, reached = _num_upsamples_to_reach(image_size, base=base_spatial)

    inputs = layers.Input(shape=(latent_dim,))
    x = inputs

    for size in cfg.get("gen_pre_dense", []):
        x = layers.Dense(size, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

    ch0 = cfg["gen_ch0"]
    x = layers.Dense(base_spatial * base_spatial * ch0, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((base_spatial, base_spatial, ch0))(x)

    for i in range(ups):
        x = layers.Conv2DTranspose(
            _filters_for_gen(i, cfg["gen_base"], cfg["gen_min"]),
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

    if reached != image_size:
        x = layers.Conv2D(
            _filters_for_gen(ups, cfg["gen_base"], cfg["gen_min"]),
            kernel_size=3,
            padding="same",
            use_bias=False,
        )(x)
        x = layers.LeakyReLU()(x)
        x = layers.Resizing(image_size, image_size, interpolation="bilinear")(x)

    if cfg.get("extra_conv", False):
        x = layers.Conv2D(32, kernel_size=3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

    outputs = layers.Conv2D(
        3, kernel_size=3, strides=1, padding="same", activation="tanh"
    )(x)
    return tf.keras.Model(inputs, outputs, name=f"Generator_{image_size}")