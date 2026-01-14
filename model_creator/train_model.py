from __future__ import annotations

import csv
import shutil
import time
from collections import defaultdict
from pathlib import Path
import random
from typing import Dict, Iterable, Mapping

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import img_to_array
from tensorflow import keras
from tqdm import tqdm

from config import (batch_size, dataset_path, latent_dimension_generator, rgb_images, sample_outputs_root_directory, save_train_epoch_every, statistics_file_path)
from ganalyzer.misc import (get_current_epoch, get_discriminator_model_path_at_given_epoch, get_generator_model_path_at_given_epoch)
from ganalyzer.models import get_discriminator, get_generator

SAMPLE_OUTPUT_PREFIX = "sample_output_epoch_"

def save_train_images(generated_images):
	for i in range(batch_size):
		this_array = generated_images[i, :, :, :]
		this_img = np.clip((this_array + 1.0) * 127.5, 0, 255).astype(np.uint8)
		filename = "subset_train/img_" + str(i) + ".png"
		Image.fromarray(this_img.astype(np.uint8), 'RGB').save(filename, format = 'PNG')
		print(f"Image saved to {filename}")

def _train_step(images, *, latent_dim, generator, discriminator, generator_optimizer, discriminator_optimizer, cross_entropy):
	noise = tf.random.normal([batch_size, latent_dim], mean = 0.0, stddev = 1.0)

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated_images = generator(noise, training = True)

		# save_train_images(generated_images.numpy())

		fake_output = discriminator(generated_images, training = True)
		real_output = discriminator(images, training = True)

		gen_loss = generator_loss(fake_output, cross_entropy)
		dis_loss = discriminator_loss(fake_output, real_output, cross_entropy)

	gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(dis_loss, discriminator.trainable_variables)

	generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

	return gen_loss, dis_loss, fake_output, real_output

def train(current_epoch, dataset, cross_entropy, latent_dim, generator, discriminator, generator_optimizer, discriminator_optimizer):
	epoch = current_epoch
	pending_statistics = []

	while True:
		print("==> current epoch : ", epoch)

		start = time.time()
		running_totals = defaultdict(float)
		batch_count = 0

		for batch in dataset:
			gen_loss, dis_loss, fake_output, real_output = _train_step(batch, latent_dim = latent_dim, generator = generator, discriminator = discriminator, generator_optimizer = generator_optimizer, discriminator_optimizer = discriminator_optimizer, cross_entropy = cross_entropy, )

			batch_stats = _collect_batch_statistics(gen_loss, dis_loss, fake_output, real_output)

			for key, value in batch_stats.items():
				running_totals[key] += value

			batch_count += 1

		time_taken = float(np.round(time.time() - start, 2))
		print("===> Time taken : ", time_taken)

		averaged_stats = _average_statistics(running_totals, batch_count)
		averaged_stats["time"] = time_taken
		pending_statistics.append((epoch, averaged_stats))

		if _should_save_models(epoch):
			_save_models(generator, discriminator, epoch, latent_dim)
			add_statistics_entries_to_file(pending_statistics)
			pending_statistics.clear()

		epoch += 1

def _should_save_models(epoch):
	return epoch == 0 or epoch % save_train_epoch_every == 0

def _save_models(generator, discriminator, epoch, latent_dim):
	print("===> saving models")
	generator.save(get_generator_model_path_at_given_epoch(epoch))
	discriminator.save(get_discriminator_model_path_at_given_epoch(epoch))
	save_generator_samples(generator, epoch, latent_dim)

def _collect_batch_statistics(gen_loss, dis_loss, fake_output, real_output):
	real_output_np = real_output.numpy()
	fake_output_np = fake_output.numpy()

	return {
		"median_real": float(np.median(real_output_np)),
		"median_fake": float(np.median(fake_output_np)),
		"mean_real": float(np.mean(real_output_np)),
		"mean_fake": float(np.mean(fake_output_np)),
		"gen_loss": float(gen_loss.numpy()),
		"disc_loss": float(dis_loss.numpy()),
	}

def _average_statistics(running_totals, batch_count):
	if batch_count == 0:
		return dict(running_totals)

	return {key: value / batch_count for key, value in running_totals.items()}

def add_statistics_entries_to_file(entries):
	if not entries:
		return

	statistics_path = Path(statistics_file_path)
	statistics_path.parent.mkdir(parents = True, exist_ok = True)

	file_exists = statistics_path.exists()
	headers = list(entries[0][1].keys())

	with statistics_path.open(mode = "a", newline = "", encoding = "utf-8") as statistics_file:
		writer = csv.writer(statistics_file)

		if not file_exists:
			writer.writerow(["epoch_id", *headers])

		for epoch, new_stats in entries:
			writer.writerow([str(epoch), *[new_stats[key] for key in headers]])

def generator_loss(fake_output, cross_entropy):
	return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(fake_output, real_output, cross_entropy):
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	return fake_loss + real_loss

def get_dataset():
	dataset_directory = Path(dataset_path)
	if not dataset_directory.exists():
		raise FileNotFoundError(f"Dataset path does not exist: {dataset_directory}")

	dataset = []

	for image_path in tqdm(sorted(dataset_directory.iterdir())):
		if not image_path.is_file():
			continue

		current_image = _load_image(image_path)
		dataset.append(img_to_array(current_image))

	if not dataset:
		raise ValueError(f"No images found in dataset path {dataset_directory}")

	return np.stack(dataset, axis = 0)

def _load_image(image_path):
	read_mode = cv2.IMREAD_COLOR if rgb_images else cv2.IMREAD_GRAYSCALE
	image = cv2.imread(str(image_path), read_mode)
	if image is None:
		raise ValueError(f"Failed to load image: {image_path}")

	if rgb_images:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image = image.astype("float32")
	return (image - 127.5) / 127.5

def save_generator_samples(generator, epoch, latent_dim, num_samples = 20):
	root_directory = Path(sample_outputs_root_directory)
	target_directory = root_directory / f"{SAMPLE_OUTPUT_PREFIX}{epoch:04d}"

	_cleanup_previous_samples(root_directory, keep = target_directory)

	if target_directory.exists():
		shutil.rmtree(target_directory)

	target_directory.mkdir(parents = True, exist_ok = True)

	print(f"===> generating sample outputs in {target_directory}")

	noise = tf.random.normal([num_samples, latent_dim], mean = 0.0, stddev = 1.0)
	generated_images = generator(noise, training = False).numpy()
	projected_images = np.clip((generated_images + 1.0) * 127.5, 0, 255).astype(np.uint8)

	for index, image_array in enumerate(projected_images):
		image = _array_to_pil_image(image_array)
		image.save(target_directory / f"sample_{index:02d}.png")

def _cleanup_previous_samples(root_directory: Path, *, keep: Path) -> None:
	if not root_directory.is_dir():
		return

	for entry in root_directory.iterdir():
		if entry == keep:
			continue
		if entry.is_dir() and entry.name.startswith(SAMPLE_OUTPUT_PREFIX):
			shutil.rmtree(entry)

def _array_to_pil_image(image_array):
	if image_array.shape[-1] == 1:
		return Image.fromarray(image_array.squeeze(-1), mode = "L")
	return Image.fromarray(image_array, mode = "RGB")

def launch_training() -> None:
	current_epoch = get_current_epoch()
	print("==> will start from epoch  : ", current_epoch)

	dataset = get_dataset()

	dataset_batches = (
		tf.data.Dataset.from_tensor_slices(dataset)
		.shuffle(buffer_size = len(dataset), reshuffle_each_iteration = True)
		.batch(batch_size)
		.prefetch(tf.data.AUTOTUNE)
	)

	if current_epoch == 0:
		print("==> Creating models")
		generator = get_generator()
		discriminator = get_discriminator()
	else:
		print("==> Loading latest models")
		discriminator = keras.models.load_model(get_discriminator_model_path_at_given_epoch(current_epoch))
		generator = keras.models.load_model(get_generator_model_path_at_given_epoch(current_epoch))

	generator.summary()
	discriminator.summary()

	generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.0001, clipvalue = 1.0)
	discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.0001, clipvalue = 1.0)

	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = False)

	cardinality = tf.data.experimental.cardinality(dataset_batches).numpy()
	if cardinality < 0:
		print("==> Number of batches : unknown")
	else:
		print(f"==> Number of batches : {int(cardinality)}")

	train(current_epoch, dataset_batches, cross_entropy, latent_dimension_generator, generator, discriminator, generator_optimizer, discriminator_optimizer)

if __name__ == "__main__":
	launch_training()
