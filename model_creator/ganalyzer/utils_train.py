from __future__ import annotations

from config import BATCH_SIZE, SAVE_TRAIN_EPOCH_EVERY, STATISTICS_FILE_PATH, LATENT_DIMENSION_GENERATOR, STR_PATH_DATASET
from ganalyzer.misc import *

import matplotlib

matplotlib.use("Agg")

from keras.preprocessing.image import img_to_array
from tensorflow import keras

import csv
from tqdm import tqdm
import time
from collections import defaultdict
from pathlib import Path

from ganalyzer.models import get_discriminator, get_generator

import numpy as np
import tensorflow as tf

def get_dataset():
	dataset_directory = Path(STR_PATH_DATASET)
	if not dataset_directory.exists():
		raise FileNotFoundError(f"Dataset path does not exist: {dataset_directory}")

	dataset = []

	for image_path in tqdm(sorted(dataset_directory.iterdir())):
		current_image = load_image(image_path)
		dataset.append(img_to_array(current_image))

	if not dataset:
		raise ValueError(f"No images found in dataset path {dataset_directory}")

	return np.stack(dataset, axis = 0)

def launch_training():
	latest_saved_epoch = get_latest_complete_checkpoint_epoch()
	start_epoch = int(latest_saved_epoch) + 1 if latest_saved_epoch is not None else 0
	print("==> latest saved epoch  : ", latest_saved_epoch if latest_saved_epoch is not None else "none")
	print("==> will start from epoch  : ", start_epoch)

	dataset = get_dataset()
	dataset_batches = build_dataset_batches(dataset)
	generator, discriminator = load_or_create_training_models(latest_saved_epoch)

	generator.summary()
	discriminator.summary()

	generator_optimizer, discriminator_optimizer = create_training_optimizers()

	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = False)

	cardinality = tf.data.experimental.cardinality(dataset_batches).numpy()
	if cardinality < 0:
		print("==> Number of batches : unknown")
	else:
		print(f"==> Number of batches : {int(cardinality)}")

	train(start_epoch, dataset_batches, cross_entropy, LATENT_DIMENSION_GENERATOR, generator, discriminator, generator_optimizer, discriminator_optimizer)

def train(start_epoch, dataset, cross_entropy, latent_dim, generator, discriminator, generator_optimizer, discriminator_optimizer):
	epoch = start_epoch
	pending_statistics = []

	while True:
		print("==> current epoch : ", epoch)

		start = time.time()
		running_totals = defaultdict(float)
		batch_count = 0

		for batch in dataset:
			gen_loss, dis_loss, fake_output, real_output = train_step(batch, latent_dim, generator, discriminator, generator_optimizer, discriminator_optimizer, cross_entropy)

			batch_stats = collect_batch_statistics(gen_loss, dis_loss, fake_output, real_output)

			for key, value in batch_stats.items():
				running_totals[key] += value

			batch_count += 1

		time_taken = float(np.round(time.time() - start, 2))
		print("===> Time taken : ", time_taken)

		averaged_stats = average_statistics(running_totals, batch_count)
		averaged_stats["time"] = time_taken
		pending_statistics.append((epoch, averaged_stats))

		if should_save_models(epoch):
			save_models(generator, discriminator, epoch)
			add_statistics_entries_to_file(pending_statistics)
			pending_statistics.clear()

		epoch += 1

def load_or_create_training_models(latest_saved_epoch):
	if latest_saved_epoch is None:
		print("==> Creating models")
		return get_generator(), get_discriminator()

	print("==> Loading latest models")
	discriminator = keras.models.load_model(get_discriminator_model_path_at_given_epoch(latest_saved_epoch))
	generator = keras.models.load_model(get_generator_model_path_at_given_epoch(latest_saved_epoch))
	return generator, discriminator

def create_training_optimizers():
	return (
		tf.keras.optimizers.RMSprop(learning_rate = 0.0001, clipvalue = 1.0),
		tf.keras.optimizers.RMSprop(learning_rate = 0.0001, clipvalue = 1.0),
	)

def build_dataset_batches(dataset):
	return (
		tf.data.Dataset.from_tensor_slices(dataset)
		.shuffle(buffer_size = len(dataset), reshuffle_each_iteration = True)
		.batch(BATCH_SIZE)
		.prefetch(tf.data.AUTOTUNE)
	)

def get_logged_statistics_epochs(statistics_path: Path):
	if not statistics_path.exists() or statistics_path.stat().st_size == 0:
		return set()

	logged_epochs = set()
	with statistics_path.open(mode = "r", newline = "", encoding = "utf-8") as statistics_file:
		reader = csv.reader(statistics_file)
		for row in reader:
			if row[0] != 'epoch_id':
				logged_epochs.add(int(row[0]))

	return logged_epochs

def add_statistics_entries_to_file(entries):
	if not entries:
		return

	statistics_path = Path(STATISTICS_FILE_PATH)
	statistics_path.parent.mkdir(parents = True, exist_ok = True)

	logged_epochs = get_logged_statistics_epochs(statistics_path)
	entries_to_write = []
	for epoch, stats in entries:
		entries_to_write.append((epoch, stats))
		logged_epochs.add(epoch)

	if not entries_to_write:
		print("===> no new statistics to write; all pending epochs are already logged")
		return

	file_has_content = statistics_path.exists() and statistics_path.stat().st_size > 0
	headers = list(entries_to_write[0][1].keys())

	with statistics_path.open(mode = "a", newline = "", encoding = "utf-8") as statistics_file:
		writer = csv.writer(statistics_file)

		if not file_has_content:
			writer.writerow(["epoch_id", *headers])

		for epoch, new_stats in entries_to_write:
			writer.writerow([str(epoch), *[new_stats[key] for key in headers]])

def save_models(generator, discriminator, epoch):
	print("===> saving models")
	generator_path = Path(get_generator_model_path_at_given_epoch(epoch))
	discriminator_path = Path(get_discriminator_model_path_at_given_epoch(epoch))
	generator_path.parent.mkdir(parents = True, exist_ok = True)
	generator.save(generator_path)
	discriminator.save(discriminator_path)

def should_save_models(epoch):
	return epoch == 0 or epoch % SAVE_TRAIN_EPOCH_EVERY == 0

def average_statistics(running_totals, batch_count):
	if batch_count == 0:
		return dict(running_totals)

	return {key: value / batch_count for key, value in running_totals.items()}

def train_step(images, latent_dim, generator, discriminator, generator_optimizer, discriminator_optimizer, cross_entropy):
	noise = tf.random.normal([BATCH_SIZE, latent_dim], mean = 0.0, stddev = 1.0)

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated_images = generator(noise, training = True)

		fake_output = discriminator(generated_images, training = True)
		real_output = discriminator(images, training = True)

		gen_loss = generator_loss(fake_output, cross_entropy)
		dis_loss = discriminator_loss(fake_output, real_output, cross_entropy)

	gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(dis_loss, discriminator.trainable_variables)

	generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

	return gen_loss, dis_loss, fake_output, real_output

def collect_batch_statistics(gen_loss, dis_loss, fake_output, real_output):
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

def generator_loss(fake_output, cross_entropy):
	return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(fake_output, real_output, cross_entropy):
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	return fake_loss + real_loss
