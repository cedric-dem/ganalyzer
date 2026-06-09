from __future__ import annotations

from config import CONTINUOUS_MOVEMENT_DIRECTORY, CONTINUOUS_MOVEMENT_LENGTH, CONTINUOUS_MOVEMENT_NUMBER_CHANGES, CONTINUOUS_MOVEMENT_IMAGE_PREFIX, EVOLUTION_SAMPLE_PATH, EVOLUTION_SAMPLE_PREFIX, GENERATOR_GLOBAL_NAME, MODEL_NAME, MODELS_DIRECTORY, SAMPLE_OUTPUT_PREFIX, IMAGE_NORMALIZATION_CENTER, \
	SAMPLE_QUANTITY, BATCH_SIZE, LATENT_DIMENSION_GENERATOR, IS_RGB_IMAGES, SAMPLE_OUTPUT_ROOT_DIRECTORY

import matplotlib
import random

from ganalyzer.misc import get_model_filename, get_last_epoch_available, parse_model_filename, get_list_of_keras_models

matplotlib.use("Agg")

import shutil
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import img_to_array
from tensorflow import keras

def produce_continuous_movement():
	generator_name = f"{MODEL_NAME}-ls_{LATENT_DIMENSION_GENERATOR:04d}"
	generate_fake_images_sample(generator_name, CONTINUOUS_MOVEMENT_LENGTH, CONTINUOUS_MOVEMENT_NUMBER_CHANGES)

def produce_evolution_sample():
	generator_name = f"{MODEL_NAME}-ls_{LATENT_DIMENSION_GENERATOR:04d}"
	save_evolution_sample_images(generator_name)

def denormalize_images(images):
	return np.clip((images + 1.0) * IMAGE_NORMALIZATION_CENTER, 0, 255).astype(np.uint8)

def save_train_images(generated_images):
	for index, generated_image in enumerate(generated_images[:BATCH_SIZE]):
		image_array = denormalize_images(generated_image)
		filename = f"subset_train/img_{index}.png"
		Image.fromarray(image_array, 'RGB').save(filename, format = 'PNG')
		print(f"Image saved to {filename}")

def prepare_output_image(image):
	if not IS_RGB_IMAGES and image.shape[-1] == 3:
		image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		image = np.expand_dims(image, axis = -1)

	output_image = denormalize_images(img_to_array(image.astype("float32")))

	if output_image.shape[-1] == 3:
		output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

	return output_image

def get_available_generator_paths():
	models_dir = Path(MODELS_DIRECTORY)
	generator_paths = []
	for model_name in get_list_of_keras_models(str(models_dir)):
		model_details = parse_model_filename(model_name)

		_epoch, model_type = model_details
		if model_type == GENERATOR_GLOBAL_NAME:
			generator_paths.append(models_dir / model_name)

	return generator_paths

def save_evolution_sample_images(generator_name):
	generator_paths = get_available_generator_paths()
	if not generator_paths:
		raise ValueError(f"No generator models available in {MODELS_DIRECTORY}.")

	print(f"Generating evolution sample for {generator_name} using {len(generator_paths)} generators")

	output_dir = Path(EVOLUTION_SAMPLE_PATH)
	output_dir.mkdir(parents = True, exist_ok = True)

	latent_vector = np.random.normal(0.0, 1.0, size = (1, LATENT_DIMENSION_GENERATOR))

	for index, generator_path in enumerate(generator_paths, start = 1):
		print(f"=> Generating image {index}/{len(generator_paths)} with {generator_path.name}")

		generator = keras.models.load_model(generator_path)
		image = generator(latent_vector, training = False).numpy()[0]
		output_image = prepare_output_image(image)

		epoch, _model_type = parse_model_filename(generator_path.name)
		output_path = output_dir / f"{EVOLUTION_SAMPLE_PREFIX}_{epoch:06d}.png"
		cv2.imwrite(str(output_path), output_image)

def produce_sample_outputs():
	generator_paths = get_available_generator_paths()
	if not generator_paths:
		raise ValueError(f"No generator checkpoints available in {MODELS_DIRECTORY}.")

	print(f"==> generating sample outputs for {len(generator_paths)} generator checkpoints")
	for index, generator_path in enumerate(generator_paths, start = 1):
		epoch, _model_type = parse_model_filename(generator_path.name)
		print(f"==> loading generator checkpoint {index}/{len(generator_paths)} from {generator_path}")
		generator = keras.models.load_model(generator_path)
		save_generator_samples(generator, epoch, LATENT_DIMENSION_GENERATOR, num_samples = SAMPLE_QUANTITY)

def generate_fake_images_sample(generator_name, length_evolution, nb_changes):
	models_dir = Path(MODELS_DIRECTORY)

	gen_epoch = get_last_epoch_available(GENERATOR_GLOBAL_NAME, str(models_dir))
	print('Generating fake images using ', generator_name, gen_epoch)

	generator_path = models_dir / get_model_filename(GENERATOR_GLOBAL_NAME, gen_epoch)
	output_dir = Path(CONTINUOUS_MOVEMENT_DIRECTORY)
	output_dir.mkdir(parents = True, exist_ok = True)

	generator = keras.models.load_model(generator_path)

	latent_vector = np.random.normal(0.0, 1.0, size = (1, LATENT_DIMENSION_GENERATOR))

	for image_index in range(length_evolution):
		print("=> Generating image ", image_index + 1, "/", length_evolution)
		print(latent_vector.shape)

		mutate_latent_vector_in_place(latent_vector, nb_changes)

		image = generator(latent_vector, training = False).numpy()[0]
		output_image = prepare_output_image(image)

		output_path = output_dir / str(CONTINUOUS_MOVEMENT_IMAGE_PREFIX + f"_{image_index + 1:04d}.png")
		cv2.imwrite(str(output_path), output_image)

def get_rnd_elem():  # could delete this ????
	return random_latent_value()

def mutate_latent_vector_in_place(latent_vector, nb_changes):
	indices = np.random.choice(latent_vector.shape[1], size = nb_changes, replace = False)
	new_values = np.random.normal(0.0, 1.0, size = nb_changes)
	latent_vector[0, indices] = new_values

def random_latent_value():
	return round(random.gauss(0, 1), 2)

def random_latent_vector(latent_space_size):
	return [random_latent_value() for _ in range(latent_space_size)]

def is_better_candidate(candidate_difference, best_difference):
	return best_difference is None or candidate_difference < best_difference

def search_random(generator, goal, ls_size, quantity_initial_random):  # returns best latent vector
	best_latent_vector = None
	best_difference = None

	for generation_index in range(quantity_initial_random):
		if (generation_index % 100) == 0:
			print("===> Search random, ", generation_index, "/", quantity_initial_random)

		candidate_vector = random_latent_vector(ls_size)
		candidate_difference = get_difference_with_original(generator, candidate_vector, goal)
		if is_better_candidate(candidate_difference, best_difference):
			print("==> new best difference : ", candidate_difference)
			best_latent_vector = candidate_vector.copy()
			best_difference = candidate_difference

	return best_latent_vector

def search_genetic_algorithm(generator, initial_latent_vector, goal, quantity_genetic_evolution, nb_diff):  # returns best latent vector
	best_latent_vector = initial_latent_vector.copy()
	best_difference = get_difference_with_original(generator, best_latent_vector, goal)

	for generation_index in range(quantity_genetic_evolution):
		if (generation_index % 100) == 0:
			print("===> Search genetic, ", generation_index, "/", quantity_genetic_evolution)

		candidate_vector = mutate_vector(initial_latent_vector, nb_diff)
		candidate_difference = get_difference_with_original(generator, candidate_vector, goal)
		if candidate_difference < best_difference:
			print("==> new best difference : ", candidate_difference)
			best_latent_vector = candidate_vector.copy()
			best_difference = candidate_difference

	return best_latent_vector

def mutate_vector(current_vector, nb_diff):
	new_vector = current_vector.copy()

	for _ in range(nb_diff):
		mutation_index = random.randint(0, len(current_vector) - 1)
		new_vector[mutation_index] = random_latent_value()

	return new_vector

def get_avg_latent_vector(all_best_latent_vectors):
	return np.mean(np.asarray(all_best_latent_vectors, dtype = np.float32), axis = 0).tolist()

def save_produced_result(generator, latent_vector, output_path):
	result = apply_model(generator, latent_vector)
	output_path.parent.mkdir(parents = True, exist_ok = True)

	cv2.imwrite(str(output_path), result)

def apply_model(generator, latent_vector):
	latent_array = np.expand_dims(np.asarray(latent_vector, dtype = "float32"), axis = 0)
	image = generator(latent_array, training = False).numpy()[0]
	return prepare_output_image(image)

def get_difference_with_original(generator, latent_vector, goal):
	reproduced_image = apply_model(generator, latent_vector)
	delta = np.abs(goal - reproduced_image)
	return delta.reshape(-1).sum()

def save_generator_samples(generator, epoch, latent_dim, num_samples):
	root_directory = Path(SAMPLE_OUTPUT_ROOT_DIRECTORY)
	target_directory = root_directory / f"{SAMPLE_OUTPUT_PREFIX}{epoch:04d}"

	if target_directory.exists():
		shutil.rmtree(target_directory)

	target_directory.mkdir(parents = True, exist_ok = True)

	print(f"===> generating sample outputs in {target_directory}")

	noise = tf.random.normal([num_samples, latent_dim], mean = 0.0, stddev = 1.0)
	generated_images = generator(noise, training = False).numpy()
	projected_images = denormalize_images(generated_images)

	for index, image_array in enumerate(projected_images):
		image = array_to_pil_image(image_array)
		image.save(target_directory / f"sample_{index:02d}.png")

def array_to_pil_image(image_array):
	if image_array.shape[-1] == 1:
		return Image.fromarray(image_array.squeeze(-1), mode = "L")
	return Image.fromarray(image_array, mode = "RGB")
