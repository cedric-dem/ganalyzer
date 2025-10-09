from ganalyzer.misc import *

import os
import csv
import time
import shutil
from tqdm import tqdm

import cv2
import keras
from keras.preprocessing.image import img_to_array
import numpy as np
from ganalyzer.models import *
import tensorflow as tf
from PIL import Image


def train(
    current_epoch,
    dataset,
    cross_entropy,
    batch_size,
    latent_dim,
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
):

    epoch = current_epoch

    @tf.function
    def train_step(images):
        noise = tf.random.normal([tf.shape(images)[0], latent_dim])

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

        return gen_loss, dis_loss, fake_output, real_output

    while True:
        print("==> current epoch : ", epoch)

        if epoch == 0 or epoch % save_train_epoch_every == 0:
            print("===> saving models")
            generator.save(get_generator_model_path_at_given_epoch(epoch))
            discriminator.save(get_discriminator_model_path_at_given_epoch(epoch))
            save_generator_samples(generator, epoch, latent_dim)

        start = time.time()

        total_stats = {}
        num_batches = 0

        for batch in dataset:
            gen_loss, dis_loss, fake_output, real_output = train_step(batch)
            num_batches += 1

            real_output_np = real_output.numpy()
            fake_output_np = fake_output.numpy()
            this_stats = {
                "median_real": float(np.median(real_output_np)),
                "median_fake": float(np.median(fake_output_np)),
                "mean_real": float(np.mean(real_output_np)),
                "mean_fake": float(np.mean(fake_output_np)),
                "gen_loss": float(gen_loss.numpy()),
                "disc_loss": float(dis_loss.numpy()),
            }

            for key in this_stats:
                if key in total_stats:
                    total_stats[key] += this_stats[key]
                else:
                    total_stats[key] = this_stats[key]

        time_taken = str(np.round(time.time() - start, 2))
        print("===> Time taken : ", time_taken)
        total_stats["time"] = time_taken

        if num_batches > 0:
            for key in list(total_stats.keys()):
                if key != "time":
                    total_stats[key] /= num_batches

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
            current_image = cv2.cvtColor(
                cv2.imread(os.path.join(dataset_path, i), cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB,
            )
        else:
            current_image = cv2.imread(os.path.join(dataset_path, i), cv2.IMREAD_GRAYSCALE)

        current_image = current_image.astype("float32")
        current_image = (current_image - 127.5) / 127.5
        dataset.append(img_to_array(current_image))

    if not dataset:
        raise ValueError(f"No images found in dataset path {dataset_path}")

    return np.stack(dataset, axis=0)


def save_generator_samples(generator, epoch, latent_dim, num_samples=20):
    current_folder_name = f"sample_output_epoch_{epoch:04d}"
    current_folder_path = os.path.join(sample_outputs_root_directory, current_folder_name)

    if os.path.isdir(sample_outputs_root_directory):
        for entry in os.listdir(sample_outputs_root_directory):
            entry_path = os.path.join(sample_outputs_root_directory, entry)
            if entry.startswith("sample_output_epoch_") and os.path.isdir(entry_path):
                if entry_path != current_folder_path:
                    shutil.rmtree(entry_path)

    if os.path.isdir(current_folder_path):
        shutil.rmtree(current_folder_path)

    os.makedirs(current_folder_path, exist_ok=True)

    print(f"===> generating sample outputs in {current_folder_path}")

    noise = tf.random.normal([num_samples, latent_dim])
    generated_images = generator(noise, training=False).numpy()
    projected_images = np.clip((generated_images + 1.0) * 127.5, 0, 255).astype(np.uint8)

    for index, image_array in enumerate(projected_images):
        if image_array.shape[-1] == 1:
            image_array = image_array.squeeze(-1)
            image = Image.fromarray(image_array, mode="L")
        else:
            image = Image.fromarray(image_array, mode="RGB")

        image.save(os.path.join(current_folder_path, f"sample_{index:02d}.png"))


def launch_training():
    current_epoch = get_current_epoch()
    print("==> will start from epoch  : ", current_epoch)

    dataset = get_dataset()

    dataset_batches = (
        tf.data.Dataset.from_tensor_slices(dataset)
        .shuffle(buffer_size=len(dataset), reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

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
    train(
        current_epoch,
        dataset_batches,
        cross_entropy,
        batch_size,
        latent_dimension_generator,
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
    )


launch_training()