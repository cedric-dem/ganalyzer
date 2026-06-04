from __future__ import annotations
import os

from config import LOAD_QUANTITY_GUI, MODELS_AS_TFLITE, NUMBER_COMPARISON, LATENT_DIMENSION_GENERATOR_AVAILABLE, STR_PATH_MODELS_ROOT, ALL_MODELS, DISCRIMINATOR_GLOBAL_NAME, GENERATOR_GLOBAL_NAME, EPOCH_GLOBAL_NAME, MODELS_DIRECTORY_NAME, LATENT_SPACE_GLOBAL_NAME, MODELS_DIRECTORY, \
	IMAGE_NORMALIZATION_CENTER, IMAGE_NORMALIZATION_SCALE, STR_PATH_DATASET, IS_RGB_IMAGES

from dataclasses import dataclass

import matplotlib
import random
from tensorflow.keras.models import load_model

matplotlib.use("Agg")

import csv
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from tensorflow import keras

def get_model_filename(model_type, epoch):
	return f"{model_type}_{EPOCH_GLOBAL_NAME}_{epoch:06d}.keras"

def normalize_image(image):
	return (image.astype("float32") - IMAGE_NORMALIZATION_CENTER) / IMAGE_NORMALIZATION_SCALE

def get_latest_complete_checkpoint_epoch():
	for epoch in sorted(get_available_epochs(MODELS_DIRECTORY), reverse = True):
		generator_path = Path(get_generator_model_path_at_given_epoch(epoch))
		discriminator_path = Path(get_discriminator_model_path_at_given_epoch(epoch))

		if generator_path.exists() and discriminator_path.exists():
			return epoch
	return None

def load_image(image_path):
	read_mode = cv2.IMREAD_COLOR if IS_RGB_IMAGES else cv2.IMREAD_GRAYSCALE
	image = cv2.imread(str(image_path), read_mode)
	if image is None:
		raise ValueError(f"Failed to load image: {image_path}")

	if IS_RGB_IMAGES:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	return normalize_image(image)

def get_model_files_directory(setting_name):
	return Path(STR_PATH_MODELS_ROOT) / setting_name / MODELS_DIRECTORY_NAME

def get_saved_model_path(setting_name, model_type, epoch_number):
	return get_model_files_directory(setting_name) / get_model_filename(model_type, epoch_number)

@dataclass
class Statistics:
	training_loss: list[float]
	validation_loss: list[float]
	generator_loss: list[float]
	discriminator_loss: list[float]
	epoch_durations: list[float]

def parse_float(value):
	return float(value.strip())

def load_statistics(csv_path):
	training_loss = []
	validation_loss = []
	generator_losses = []
	discriminator_losses = []
	epoch_durations = []

	with csv_path.open(newline = "", encoding = "utf-8") as csv_file:
		reader = csv.DictReader(csv_file)
		for row in reader:
			for key, raw_value in row.items():
				value = parse_float(raw_value)

				lower_key = key.lower()
				if "gen" in lower_key and "loss" in lower_key:
					generator_losses.append(value)
				elif "disc" in lower_key and "loss" in lower_key:
					discriminator_losses.append(value)
				elif "val_loss" in lower_key or lower_key == "validation_loss":
					validation_loss.append(value)
				elif lower_key == "loss" or lower_key.endswith("_loss"):
					training_loss.append(value)
				elif "time" in lower_key or "duration" in lower_key:
					epoch_durations.append(value)

	return Statistics(
		training_loss = training_loss,
		validation_loss = validation_loss,
		generator_loss = generator_losses,
		discriminator_loss = discriminator_losses,
		epoch_durations = epoch_durations,
	)

def parse_setting_name(setting_name):
	model_size, latent_space_name = setting_name.rsplit("-", 1)

	latent_space_size = int(latent_space_name.removeprefix(LATENT_SPACE_GLOBAL_NAME + "_"))

	return model_size, latent_space_size

def is_configured_setting(setting_name):
	parsed_setting = parse_setting_name(setting_name)
	if parsed_setting is None:
		return False

	model_size, latent_space_size = parsed_setting
	return model_size in ALL_MODELS and latent_space_size in LATENT_DIMENSION_GENERATOR_AVAILABLE

def get_model_indexes(model_name):
	parsed_setting = parse_setting_name(model_name)
	if parsed_setting is None:
		raise ValueError(f"Invalid model setting name: {model_name}")

	model_size, latent_size = parsed_setting
	idx_x = ALL_MODELS.index(model_size)
	idx_y = LATENT_DIMENSION_GENERATOR_AVAILABLE.index(latent_size)

	return idx_x, idx_y

def get_real_images_sample():
	print('getting real images ')
	dataset_directory = Path(STR_PATH_DATASET)

	image_paths = [path for path in sorted(dataset_directory.iterdir()) if path.is_file()]

	selected_paths = random.sample(image_paths, k = min(NUMBER_COMPARISON, len(image_paths)))
	images = []

	for image_path in selected_paths:
		normalized_image = load_image(image_path)

		images.append(img_to_array(normalized_image))

	return images

def get_fake_images_sample(generator_name, generator_epoch):
	print('Generating fake images using ', generator_name, generator_epoch)
	epoch_number = int(str(generator_epoch).replace(str(EPOCH_GLOBAL_NAME + "_"), ""))

	generator_path = get_saved_model_path(generator_name, GENERATOR_GLOBAL_NAME, epoch_number)

	generator = keras.models.load_model(generator_path)
	ls_size = int(generator_name.split("_")[-1])
	latent_vectors = np.random.normal(0.0, 1.0, size = (NUMBER_COMPARISON, ls_size))

	generated_images = generator(latent_vectors, training = False).numpy()
	images = []

	for image in generated_images:
		if not IS_RGB_IMAGES and image.shape[-1] == 3:
			image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			image = np.expand_dims(image, axis = -1)

		images.append(img_to_array(image.astype("float32")))

	return images

def get_accuracy_on_images(model_name, model_epoch, images_set, is_real_images):
	# could use get_model_path_at_given_epoch(model_type, current_best_result) ? I don't know
	# should obtain the accuracy of a given model on a given epoch, using a special set of images. they are either all real or all fake, given by boolean is_real_images

	epoch_number = int(str(model_epoch).replace(str(EPOCH_GLOBAL_NAME + "_"), ""))

	model_path = get_saved_model_path(model_name, DISCRIMINATOR_GLOBAL_NAME, epoch_number)

	discriminator = keras.models.load_model(model_path)
	images_array = np.asarray(images_set, dtype = np.float32)
	predictions = np.squeeze(discriminator(images_array, training = False).numpy())

	expected_label = 1.0 if is_real_images else 0.0
	predicted_labels = (predictions >= 0.5).astype(np.float32)

	return float(np.mean(predicted_labels == expected_label))

def parse_model_filename(filename):
	parts = Path(filename).stem.split("_")
	return int(parts[-1]), "_".join(parts[:-2])

def get_number_epoch_in_given_setting(setting):
	setting_models_directory = get_model_files_directory(setting)

	max_epoch = 0

	for model_file in setting_models_directory.iterdir():
		model_details = parse_model_filename(model_file.name)
		current_epoch, _model_type = model_details
		max_epoch = max(max_epoch, current_epoch)

	return max_epoch

def get_epoch_name(current_epoch):  # TODO use this in train etc
	return EPOCH_GLOBAL_NAME + "_" + ((6 - len(str(current_epoch))) * "0") + str(current_epoch)

def get_ls_name(current_latent_dimension_generator):
	return LATENT_SPACE_GLOBAL_NAME + "_" + ((4 - len(str(current_latent_dimension_generator))) * "0") + str(current_latent_dimension_generator)

def load_keras_model(model_path):
	return tf.keras.models.load_model(str(model_path), compile = False)

def build_concrete_function(model):
	input_specs = [tf.TensorSpec(shape = [dim if dim is not None else 1 for dim in tensor.shape], dtype = tensor.dtype) for tensor in model.inputs]

	@tf.function
	def model_fn(*args):
		return model(*args, training = False)

	return model_fn.get_concrete_function(*input_specs)

def configure_converter(concrete_fn, model):
	converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], model)
	converter.experimental_new_converter = False
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
	converter.allow_custom_ops = False
	converter.optimizations = []
	return converter

def export_tflite(model_path_keras, model_path_tflite):
	model = load_keras_model(model_path_keras)
	concrete_function = build_concrete_function(model)
	converter = configure_converter(concrete_function, model)
	tflite_model = converter.convert()

	target_path = Path(model_path_tflite)
	target_path.parent.mkdir(parents = True, exist_ok = True)
	target_path.write_bytes(tflite_model)

def default_models():
	last_generator = None
	last_discriminator = None

	for model_name in get_list_of_keras_models(MODELS_DIRECTORY):
		model_details = parse_model_filename(model_name)

		_epoch, model_type = model_details
		model_path = Path(MODELS_DIRECTORY) / model_name
		if model_type == DISCRIMINATOR_GLOBAL_NAME:
			last_discriminator = model_path
		elif model_type == GENERATOR_GLOBAL_NAME:
			last_generator = model_path

	results = []
	if last_generator is not None:
		results.append((last_generator, Path(MODELS_AS_TFLITE) / str(GENERATOR_GLOBAL_NAME + ".tflite")))
	if last_discriminator is not None:
		results.append((last_discriminator, Path(MODELS_AS_TFLITE) / str(DISCRIMINATOR_GLOBAL_NAME + ".tflite")))
	return results

def convert_keras_to_tflite():
	for source, target in default_models():
		print(f"Converting {source} -> {target}")
		export_tflite(source, target)

def model_directory_for(model_name, latent_space_size):
	return os.path.join(
		Path(STR_PATH_MODELS_ROOT),
		f"{model_name}-ls_{latent_space_size:04d}",
		MODELS_DIRECTORY_NAME,
	)

def get_model_path_at_given_epoch(model_type, epoch, models_dir):
	return os.path.join(models_dir, get_model_filename(model_type, epoch))

def get_generator_model_path_at_given_epoch(epoch):
	return get_model_path_at_given_epoch("generator", epoch, MODELS_DIRECTORY)

def get_discriminator_model_path_at_given_epoch(epoch):
	return get_model_path_at_given_epoch("discriminator", epoch, MODELS_DIRECTORY)

def get_model_path_at_given_epoch_closest_possible(model_type, epoch, available_epochs, models_dir):
	current_best_distance = None
	current_best_result = None

	for available_epoch in available_epochs:
		this_distance = abs(available_epoch - epoch)
		if current_best_distance is None or current_best_distance > this_distance:
			current_best_distance = this_distance
			current_best_result = available_epoch

	if current_best_result is None:
		raise ValueError("No available epochs supplied.")

	return get_model_path_at_given_epoch(model_type, current_best_result, models_dir)

def get_available_epochs(models_dir):
	models_list = get_list_of_keras_models(models_dir)
	return [epoch for epoch, model_type in filter(None, (parse_model_filename(model) for model in models_list)) if model_type == DISCRIMINATOR_GLOBAL_NAME]

def indexes_to_load(models_quantity):
	if models_quantity == 0:
		return []

	if LOAD_QUANTITY_GUI >= models_quantity:
		return list(range(models_quantity))

	target_count = min(LOAD_QUANTITY_GUI, models_quantity)
	if target_count == 1:
		return [0]

	step = (models_quantity - 1) / (target_count - 1)
	indexes = [int(step * index) for index in range(target_count)]
	indexes[-1] = models_quantity - 1
	res = sorted(set(indexes))
	return res

def get_all_models(model_type, available_epochs, model_name, latent_space_size):
	models_dir = model_directory_for(model_name, latent_space_size)

	models_quantity = get_current_epoch(models_dir)
	indexes = indexes_to_load(models_quantity)

	result = [None for _ in range(models_quantity)]

	for current_index in indexes:
		filename = get_model_path_at_given_epoch_closest_possible(
			model_type,
			current_index,
			available_epochs,
			models_dir,
		)
		print(f"=> will load {model_type} epoch {current_index}, "f"closest found is : {filename}")
		result[current_index] = keras.models.load_model(filename)

	return result

def project_array(arr, destination_max, project_from, project_to):
	delta = project_to - project_from
	if delta > 0:
		return ((arr - project_from) / delta) * destination_max
	return arr

def get_list_of_keras_models(models_dir):
	if not os.path.isdir(models_dir):
		return []

	complete_list = sorted(os.listdir(models_dir))
	return [filename for filename in complete_list]  # if not filename.endswith(".csv")

def get_current_epoch(models_dir):
	keras_models = get_list_of_keras_models(models_dir)
	available_epochs = [epoch for epoch, _model_type in filter(None, (parse_model_filename(model) for model in keras_models))]
	return max(available_epochs)

def get_last_epoch_available(model_type, models_dir):
	models_list = get_list_of_keras_models(models_dir)
	candidates = [
		epoch
		for epoch, current_model_type in filter(None, (parse_model_filename(model) for model in models_list))
		if current_model_type == model_type
	]

	if not candidates:
		raise ValueError(f"No {model_type} models available in {models_dir}.")

	return max(candidates)
