from misc import *
from models import *

import os
import csv
import time
from tqdm import tqdm

import cv2
import keras
from keras.preprocessing.image import img_to_array
import numpy as np
import tensorflow as tf
import random

def train(current_epoch, dataset, cross_entropy, batch_size, latent_dim, generator, discriminator, generator_optimizer, discriminator_optimizer):

    epoch = current_epoch
    while True:
        print("==> current epoch : ", epoch)

        if epoch == 0 or epoch % save_train_epoch_every == 0:
            print("===> saving models")
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

        time_taken = str(np.round(time.time() - start, 2))
        print("===> Time taken : ", time_taken)
        total_stats["time"] = time_taken

        # TODO fix csv update possibly overlapping old train
        add_statistics_to_file(epoch, total_stats)
        epoch += 1

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
    dataset = []

    files = os.listdir(dataset_path)
    for i in tqdm(files):
        if rgb_images:
            current_image = cv2.cvtColor(cv2.imread(os.path.join(dataset_path, i), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        else:
            current_image = cv2.imread(os.path.join(dataset_path, i), cv2.IMREAD_GRAYSCALE)

        # resizing image
        # current_image = cv2.resize(current_image, (SIZE, SIZE))
        current_image = (current_image - 127.5) / 127.5
        dataset.append(img_to_array(current_image))
    return dataset


def launch_training():
    current_epoch = get_current_epoch()
    print("==> will start from epoch  : ", current_epoch)

    dataset = get_dataset()

    dataset_batches = tf.data.Dataset.from_tensor_slices(np.array(dataset)).shuffle(buffer_size=len(dataset), reshuffle_each_iteration=True).batch(batch_size)

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

    print("==> Number of batches : ", len(dataset_batches))
    train(current_epoch, dataset_batches, cross_entropy, batch_size, latent_dimension_generator, generator, discriminator, generator_optimizer, discriminator_optimizer)


launch_training()
