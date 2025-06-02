from misc import *

import os
import csv
import time
from tqdm import tqdm

import cv2
import keras
from keras import layers
from keras.preprocessing.image import img_to_array
import numpy as np
import tensorflow as tf


def train(current_epoch, dataset, cross_entropy, batch_size, latent_dim, generator, discriminator, generator_optimizer, discriminator_optimizer):

    for epoch in range(current_epoch, 999):
        print("==> current epoch : ", epoch)

        generator.save(get_generator_model_path_at_given_epoch(epoch))
        discriminator.save(get_discriminator_model_path_at_given_epoch(epoch))

        start = time.time()

        total_stats = {}

        for batch in dataset:
            this_stats = train_steps(batch, cross_entropy, batch_size, latent_dim, generator, discriminator, generator_optimizer, discriminator_optimizer)

            for key in this_stats:
                if key in total_stats:
                    total_stats[key] += this_stats[key]
                else:
                    total_stats[key] = 0

        total_stats["time"] = str(np.round(time.time() - start, 2))

        add_statistics_to_file(epoch, total_stats)


def add_statistics_to_file(epoch, new_stats):

    exists = os.path.isfile(statistics_file_path)

    with open(statistics_file_path, mode="a", newline="", encoding="utf-8") as statistics_file:
        writer = csv.writer(statistics_file)

        if not exists:
            writer.writerow(["epoch_id"] + [key for key in new_stats])

        writer.writerow([str(epoch)] + [new_stats[key] for key in new_stats])


def train_steps(images, cross_entropy, batch_size, latent_dim, generator, discriminator, generator_optimizer, discriminator_optimizer):
    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        fake_output = discriminator(generated_images, training=True)
        real_output = discriminator(images, training=True)

        gen_loss = generator_loss(fake_output, cross_entropy)
        dis_loss = discriminator_loss(fake_output, real_output, cross_entropy)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(dis_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return {"median_real": np.median(real_output), "median_fake": np.median(fake_output), "mean_real": np.mean(real_output), "mean_fake": np.mean(fake_output), "gen_loss": gen_loss.numpy(), "disc_loss": dis_loss.numpy()}


def generator_loss(fake_output, cross_entropy):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(fake_output, real_output, cross_entropy):
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    return fake_loss + real_loss


def get_dataset():
    _img = []

    files = os.listdir(dataset_path)
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


def launch_training():
    current_epoch = get_current_epoch()
    print("==> will start from epoch  : ", current_epoch)

    _img = get_dataset()

    batch_size = 32
    dataset = tf.data.Dataset.from_tensor_slices(np.array(_img)).shuffle(buffer_size=len(_img), reshuffle_each_iteration=True).batch(batch_size)

    if current_epoch == 0:  # if start from scratch
        print("==> Creating models")
        generator = get_generator()
        discriminator = get_discriminator()

    else:
        print("==> Loading latest models")

        discriminator = keras.models.load_model(get_discriminator_model_path_at_given_epoch(current_epoch))
        generator = keras.models.load_model(get_generator_model_path_at_given_epoch(current_epoch))

    generator.summary()
    discriminator.summary()

    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001, clipvalue=1.0)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001, clipvalue=1.0)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    print("==> Number of batches : ", len(dataset))
    train(current_epoch, dataset, cross_entropy, batch_size, latent_dimension_generator, generator, discriminator, generator_optimizer, discriminator_optimizer)


launch_training()
