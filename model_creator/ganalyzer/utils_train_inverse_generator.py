from __future__ import annotations

from config import (
	BATCH_SIZE,
	EPOCH_GLOBAL_NAME,
	GENERATOR_GLOBAL_NAME,
	INVERSE_GENERATOR_DIRECTORY_NAME,
	INVERSE_GENERATOR_MODEL_TYPE,
	INVERSE_GENERATOR_TRAIN_EPOCHS,
	LATENT_DIMENSION_GENERATOR,
	MODELS_DIRECTORY,
	MODELS_DIRECTORY_NAME,
	MODELS_DIRECTORY_NAME_INVERSE,
	STATISTICS_CSV_FILENAME,
	STEPS_PER_INVERSE_GENERATOR_EPOCH,
	SAVE_TRAIN_EPOCH_EVERY, INVERSE_GENERATOR_STATISTICS_HEADERS,
)
from ganalyzer.misc import get_model_filename, get_model_path_at_given_epoch
from ganalyzer.models import get_inverse_generator

from pathlib import Path
import csv
import shutil
import time

from tensorflow import keras
import numpy as np
import tensorflow as tf

def launch_inverse_generator_training(train_epochs = INVERSE_GENERATOR_TRAIN_EPOCHS):
	train_epochs = int(train_epochs)
	steps_per_epoch = int(STEPS_PER_INVERSE_GENERATOR_EPOCH)
	generator_epoch = get_last_model_epoch(GENERATOR_GLOBAL_NAME, MODELS_DIRECTORY)
	generator_path = get_model_path_at_given_epoch(GENERATOR_GLOBAL_NAME, generator_epoch, MODELS_DIRECTORY)
	inverse_generator_path = get_inverse_generator_model_path(generator_epoch, train_epochs)

	delete_previous_inverse_generator_outputs(generator_epoch)

	print("==> loading generator from : ", generator_path)
	generator = keras.models.load_model(str(generator_path))
	generator.trainable = False

	inverse_generator = load_or_create_inverse_generator(inverse_generator_path)
	inverse_generator.summary()

	optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.0001, clipvalue = 1.0)
	loss_fn = tf.keras.losses.MeanSquaredError()

	train_inverse_generator(generator, inverse_generator, optimizer, loss_fn, generator_epoch, train_epochs, steps_per_epoch)

def delete_previous_inverse_generator_outputs(generator_epoch):
	models_directory = get_inverse_generator_models_directory(generator_epoch)
	statistics_path = get_inverse_generator_statistics_path(generator_epoch)

	if models_directory.exists():
		print("==> deleting previous inverse generator models directory : ", models_directory)
		shutil.rmtree(models_directory)

	if statistics_path.exists():
		print("==> deleting previous inverse generator statistics file : ", statistics_path)
		statistics_path.unlink()

def train_inverse_generator(generator, inverse_generator, optimizer, loss_fn, generator_epoch, train_epochs, steps_per_epoch):
	if train_epochs < 1:
		raise ValueError("train_epochs must be at least 1")
	if steps_per_epoch < 1:
		raise ValueError("steps_per_epoch must be at least 1")

	start = time.time()

	for epoch in range(1, train_epochs + 1):
		epoch_start = time.time()
		running_loss_mse = 0.0
		running_loss_mae = 0.0

		for step in range(1, steps_per_epoch + 1):
			step_loss_mse, step_loss_mae = train_inverse_generator_step(generator, inverse_generator, optimizer, loss_fn)
			loss_mse = float(step_loss_mse.numpy())
			loss_mae = float(step_loss_mae.numpy())
			running_loss_mse += loss_mse
			running_loss_mae += loss_mae

		time_taken = float(np.round(time.time() - epoch_start, 2))
		loss_mse = running_loss_mse / steps_per_epoch
		loss_mae = running_loss_mae / steps_per_epoch
		add_inverse_generator_statistics_entry(generator_epoch, epoch, time_taken, loss_mse, loss_mae)
		print(f"==> epoch {epoch:04d}/{train_epochs:04d} - step {steps_per_epoch:04d}/{steps_per_epoch:04d} complete - loss_mse : {loss_mse:.6f} - loss_mae : {loss_mae:.6f}")

		if should_save_inverse_generator(epoch, train_epochs):
			save_inverse_generator(inverse_generator, get_inverse_generator_model_path(generator_epoch, epoch))

	print("==> inverse generator training time : ", float(np.round(time.time() - start, 2)))

@tf.function
def train_inverse_generator_step(generator, inverse_generator, optimizer, loss_fn):
	latent_vectors = tf.random.normal([BATCH_SIZE, LATENT_DIMENSION_GENERATOR], mean = 0.0, stddev = 1.0)
	generated_images = generator(latent_vectors, training = False)

	with tf.GradientTape() as tape:
		predicted_latent_vectors = inverse_generator(generated_images, training = True)
		loss_mse = loss_fn(latent_vectors, predicted_latent_vectors)

	gradients = tape.gradient(loss_mse, inverse_generator.trainable_variables)
	optimizer.apply_gradients(zip(gradients, inverse_generator.trainable_variables))

	reconstructed_images = generator(predicted_latent_vectors, training = False)
	loss_mae = tf.reduce_mean(tf.abs(generated_images - reconstructed_images))
	return loss_mse, loss_mae

def add_inverse_generator_statistics_entry(generator_epoch, inverse_generator_epoch, time_taken, loss_mse, loss_mae):
	statistics_path = get_inverse_generator_statistics_path(generator_epoch)
	statistics_path.parent.mkdir(parents = True, exist_ok = True)
	file_has_content = statistics_path.exists() and statistics_path.stat().st_size > 0

	with statistics_path.open(mode = "a", newline = "", encoding = "utf-8") as statistics_file:
		writer = csv.writer(statistics_file)

		if not file_has_content:
			writer.writerow(INVERSE_GENERATOR_STATISTICS_HEADERS)

		writer.writerow([inverse_generator_epoch, time_taken, round(loss_mse, 4), round(loss_mae, 4)])

def get_inverse_generator_statistics_path(generator_epoch):
	return get_inverse_generator_run_directory(generator_epoch) / STATISTICS_CSV_FILENAME

def should_save_inverse_generator(epoch, train_epochs):
	return epoch == train_epochs or epoch % SAVE_TRAIN_EPOCH_EVERY == 0 or epoch == 1

def load_or_create_inverse_generator(inverse_generator_path):
	path = Path(inverse_generator_path)
	if path.exists():
		print("==> loading inverse generator from : ", inverse_generator_path)
		return keras.models.load_model(str(path))

	print("==> creating inverse generator")
	return get_inverse_generator()

def save_inverse_generator(inverse_generator, inverse_generator_path):
	print("==> saving inverse generator to : ", inverse_generator_path)
	path = Path(inverse_generator_path)
	path.parent.mkdir(parents = True, exist_ok = True)
	inverse_generator.save(path)

def get_inverse_generator_model_path(generator_epoch, inverse_generator_epoch):
	return get_inverse_generator_models_directory(generator_epoch) / get_model_filename(INVERSE_GENERATOR_MODEL_TYPE, inverse_generator_epoch)

def get_inverse_generator_models_directory(generator_epoch):
	return get_inverse_generator_run_directory(generator_epoch) / MODELS_DIRECTORY_NAME_INVERSE

def get_inverse_generator_run_directory(generator_epoch):
	return Path(MODELS_DIRECTORY).parent / INVERSE_GENERATOR_DIRECTORY_NAME / f"{MODELS_DIRECTORY_NAME}_{INVERSE_GENERATOR_MODEL_TYPE}_{generator_epoch}"

def get_model_filename(model_type, epoch):
	return f"{model_type}_{EPOCH_GLOBAL_NAME}_{epoch:06d}.keras"

def get_model_path_at_given_epoch(model_type, epoch, models_dir):
	return Path(models_dir) / get_model_filename(model_type, epoch)

def parse_model_filename(filename):
	path = Path(filename)
	if path.suffix != ".keras":
		return None

	parts = path.stem.split("_")
	if len(parts) < 3:
		return None

	try:
		epoch = int(parts[-1])
	except ValueError:
		return None

	model_type = "_".join(parts[:-2])
	return epoch, model_type

def get_last_model_epoch(model_type, models_dir):
	models_path = Path(models_dir)
	if not models_path.is_dir():
		raise ValueError(f"Models directory does not exist: {models_path}")

	candidates = []
	for model_path in models_path.iterdir():
		model_details = parse_model_filename(model_path.name)
		if model_details is None:
			continue

		epoch, current_model_type = model_details
		if current_model_type == model_type:
			candidates.append(epoch)

	if not candidates:
		raise ValueError(f"No {model_type} models available in {models_path}.")

	return max(candidates)
