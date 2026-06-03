from __future__ import annotations
import os

from config import LOAD_QUANTITY_GUI, MODELS_AS_TFLITE, CONTINUOUS_MOVEMENT_DIRECTORY, CONTINUOUS_MOVEMENT_LENGTH, CONTINUOUS_MOVEMENT_NUMBER_CHANGES, CONTINUOUS_MOVEMENT_IMAGE_PREFIX, REPRODUCED_IMAGES_OUTPUT_DIRECTORY, IMAGE_TO_REPRODUCE, REPRODUCED_IMAGE_SUFFIX, EVOLUTION_SAMPLE_PATH, \
	EVOLUTION_SAMPLE_PREFIX, PLOTS_ROOT_DIRECTORY, RESULTS_ROOT_PATH, NUMBER_COMPARISON, LATENT_DIMENSION_GENERATOR_AVAILABLE, MODELS_ROOT_PATH, NUMBER_EPOCH_TAKEN_COMPARISON, PLOTS_HEATMAP_EPOCHS_DIRECTORY, PLOTS_HEATMAP_MODEL_SIZE_DIRECTORY, PLOTS_HEATMAP_LATENT_SPACE_SIZE_DIRECTORY, \
	PATH_LOSS_PLOTS, PATH_LOSS_BY_LS_PLOTS, PATH_LOSS_BY_MODEL_PLOTS, PLOTS_NUMBER_PARAMETERS_DIRECTORY, ALL_MODELS, DISCRIMINATOR_GLOBAL_NAME, GENERATOR_GLOBAL_NAME, EPOCH_GLOBAL_NAME, MODELS_DIRECTORY_NAME, PLOT_IMAGE_NAMES, PLOT_NAMES, LATENT_SPACE_GLOBAL_NAME, X_LABEL_NAMES, Y_LABEL_NAMES, \
	STATISTICS_CSV_FILENAME, MODEL_NAME, MODELS_DIRECTORY, QUANTITY_INITIAL_RANDOM, QUANTITY_GENETIC_EVO, QUANTITY_GENETIC_ALGO, NB_RETRIES_AVG, SAMPLE_OUTPUT_PREFIX

from dataclasses import dataclass

import matplotlib
import colorsys
import random
import statistics
from tensorflow.keras.models import load_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import csv
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import img_to_array
from tensorflow import keras
from tqdm import tqdm

from config import (BATCH_SIZE, DATASET_PATH, LATENT_DIMENSION_GENERATOR, IS_RGB_IMAGES, SAMPLE_OUTPUT_ROOT_DIRECTORY, SAVE_TRAIN_EPOCH_EVERY, STATISTICS_FILE_PATH)
from ganalyzer.models import get_discriminator, get_generator

IMAGE_NORMALIZATION_CENTER = 127.5
IMAGE_NORMALIZATION_SCALE = 127.5

PLOTS_ROOT_DIRECTORY_PATH = Path(PLOTS_ROOT_DIRECTORY)
RESULTS_ROOT_PATH = Path(RESULTS_ROOT_PATH)
MODELS_ROOT_PATH = Path(MODELS_ROOT_PATH)
DATASET_PATH = Path(DATASET_PATH)
PLOTS_NUMBER_PARAMETERS_PATH = Path(PLOTS_NUMBER_PARAMETERS_DIRECTORY)
PLOTS_HEATMAP_EPOCHS_PATH = Path(PLOTS_HEATMAP_EPOCHS_DIRECTORY)
PLOTS_HEATMAP_MODEL_SIZE_PATH = Path(PLOTS_HEATMAP_MODEL_SIZE_DIRECTORY)
PLOTS_HEATMAP_LATENT_SPACE_SIZE_PATH = Path(PLOTS_HEATMAP_LATENT_SPACE_SIZE_DIRECTORY)
PATH_LOSS_PLOTS_PATH = Path(PATH_LOSS_PLOTS)
PATH_LOSS_PLOTS_BY_LS_PATH = Path(PATH_LOSS_BY_LS_PLOTS)
PATH_LOSS_PLOTS_BY_MODEL_PATH = Path(PATH_LOSS_BY_MODEL_PLOTS)

def get_model_filename(model_type, epoch):
	return f"{model_type}_{EPOCH_GLOBAL_NAME}_{epoch:06d}.keras"

def parse_model_filename(filename):
	parts = Path(filename).stem.split("_")
	return int(parts[-1]), "_".join(parts[:-2])

def _denormalize_images(images):
	return np.clip((images + 1.0) * IMAGE_NORMALIZATION_CENTER, 0, 255).astype(np.uint8)

def _normalize_image(image):
	return (image.astype("float32") - IMAGE_NORMALIZATION_CENTER) / IMAGE_NORMALIZATION_SCALE

def save_train_images(generated_images):
	for index, generated_image in enumerate(generated_images[:BATCH_SIZE]):
		image_array = _denormalize_images(generated_image)
		filename = f"subset_train/img_{index}.png"
		Image.fromarray(image_array, 'RGB').save(filename, format = 'PNG')
		print(f"Image saved to {filename}")

def _train_step(images, *, latent_dim, generator, discriminator, generator_optimizer, discriminator_optimizer, cross_entropy):
	noise = tf.random.normal([BATCH_SIZE, latent_dim], mean = 0.0, stddev = 1.0)

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

def train(start_epoch, dataset, cross_entropy, latent_dim, generator, discriminator, generator_optimizer, discriminator_optimizer):
	epoch = start_epoch
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
	return epoch == 0 or epoch % SAVE_TRAIN_EPOCH_EVERY == 0

def _get_latest_complete_checkpoint_epoch():
	for epoch in sorted(get_available_epochs(), reverse = True):
		generator_path = Path(get_generator_model_path_at_given_epoch(epoch))
		discriminator_path = Path(get_discriminator_model_path_at_given_epoch(epoch))

		if generator_path.exists() and discriminator_path.exists():
			return epoch

def _save_models(generator, discriminator, epoch, latent_dim):
	print("===> saving models")
	generator_path = Path(get_generator_model_path_at_given_epoch(epoch))
	discriminator_path = Path(get_discriminator_model_path_at_given_epoch(epoch))
	generator_path.parent.mkdir(parents = True, exist_ok = True)
	generator.save(generator_path)
	discriminator.save(discriminator_path)

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

	statistics_path = Path(STATISTICS_FILE_PATH)
	statistics_path.parent.mkdir(parents = True, exist_ok = True)

	logged_epochs = _get_logged_statistics_epochs(statistics_path)
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

def _get_logged_statistics_epochs(statistics_path: Path):
	if not statistics_path.exists() or statistics_path.stat().st_size == 0:
		return set()

	logged_epochs = set()
	with statistics_path.open(mode = "r", newline = "", encoding = "utf-8") as statistics_file:
		reader = csv.reader(statistics_file)
		for row in reader:
			logged_epochs.add(int(row[0]))

	return logged_epochs

def generator_loss(fake_output, cross_entropy):
	return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(fake_output, real_output, cross_entropy):
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	return fake_loss + real_loss

def get_dataset():
	dataset_directory = Path(DATASET_PATH)
	if not dataset_directory.exists():
		raise FileNotFoundError(f"Dataset path does not exist: {dataset_directory}")

	dataset = []

	for image_path in tqdm(sorted(dataset_directory.iterdir())):
		current_image = _load_image(image_path)
		dataset.append(img_to_array(current_image))

	if not dataset:
		raise ValueError(f"No images found in dataset path {dataset_directory}")

	return np.stack(dataset, axis = 0)

def _load_image(image_path):
	read_mode = cv2.IMREAD_COLOR if IS_RGB_IMAGES else cv2.IMREAD_GRAYSCALE
	image = cv2.imread(str(image_path), read_mode)
	if image is None:
		raise ValueError(f"Failed to load image: {image_path}")

	if IS_RGB_IMAGES:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	return _normalize_image(image)

def save_generator_samples(generator, epoch, latent_dim, num_samples = 20, cleanup_previous = True):
	root_directory = Path(SAMPLE_OUTPUT_ROOT_DIRECTORY)
	target_directory = root_directory / f"{SAMPLE_OUTPUT_PREFIX}{epoch:04d}"

	if cleanup_previous:
		_cleanup_previous_samples(root_directory, keep = target_directory)

	if target_directory.exists():
		shutil.rmtree(target_directory)

	target_directory.mkdir(parents = True, exist_ok = True)

	print(f"===> generating sample outputs in {target_directory}")

	noise = tf.random.normal([num_samples, latent_dim], mean = 0.0, stddev = 1.0)
	generated_images = generator(noise, training = False).numpy()
	projected_images = _denormalize_images(generated_images)

	for index, image_array in enumerate(projected_images):
		image = _array_to_pil_image(image_array)
		image.save(target_directory / f"sample_{index:02d}.png")

def _cleanup_previous_samples(root_directory: Path, *, keep: Path):
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

def produce_sample_outputs(num_samples = 20):
	generator_paths = _get_available_generator_paths()
	if not generator_paths:
		raise ValueError(f"No generator checkpoints available in {MODELS_DIRECTORY}.")

	root_directory = Path(SAMPLE_OUTPUT_ROOT_DIRECTORY)
	_cleanup_previous_samples(root_directory, keep = root_directory)

	print(f"==> generating sample outputs for {len(generator_paths)} generator checkpoints")
	for index, generator_path in enumerate(generator_paths, start = 1):
		epoch, _model_type = parse_model_filename(generator_path.name)
		print(f"==> loading generator checkpoint {index}/{len(generator_paths)} from {generator_path}")
		generator = keras.models.load_model(generator_path)
		save_generator_samples(generator, epoch, LATENT_DIMENSION_GENERATOR, num_samples = num_samples, cleanup_previous = False)

def _build_dataset_batches(dataset):
	return (
		tf.data.Dataset.from_tensor_slices(dataset)
		.shuffle(buffer_size = len(dataset), reshuffle_each_iteration = True)
		.batch(BATCH_SIZE)
		.prefetch(tf.data.AUTOTUNE)
	)

def _load_or_create_training_models(latest_saved_epoch):
	if latest_saved_epoch is None:
		print("==> Creating models")
		return get_generator(), get_discriminator()

	print("==> Loading latest models")
	discriminator = keras.models.load_model(get_discriminator_model_path_at_given_epoch(latest_saved_epoch))
	generator = keras.models.load_model(get_generator_model_path_at_given_epoch(latest_saved_epoch))
	return generator, discriminator

def _create_training_optimizers():
	return (
		tf.keras.optimizers.RMSprop(learning_rate = 0.0001, clipvalue = 1.0),
		tf.keras.optimizers.RMSprop(learning_rate = 0.0001, clipvalue = 1.0),
	)

def launch_training():
	latest_saved_epoch = _get_latest_complete_checkpoint_epoch()
	start_epoch = int(latest_saved_epoch) + 1 if latest_saved_epoch is not None else 0
	print("==> latest saved epoch  : ", latest_saved_epoch if latest_saved_epoch is not None else "none")
	print("==> will start from epoch  : ", start_epoch)

	dataset = get_dataset()
	dataset_batches = _build_dataset_batches(dataset)
	generator, discriminator = _load_or_create_training_models(latest_saved_epoch)

	generator.summary()
	discriminator.summary()

	generator_optimizer, discriminator_optimizer = _create_training_optimizers()

	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = False)

	cardinality = tf.data.experimental.cardinality(dataset_batches).numpy()
	if cardinality < 0:
		print("==> Number of batches : unknown")
	else:
		print(f"==> Number of batches : {int(cardinality)}")

	train(start_epoch, dataset_batches, cross_entropy, LATENT_DIMENSION_GENERATOR, generator, discriminator, generator_optimizer, discriminator_optimizer)

def _get_model_files_directory(setting_name):
	return MODELS_ROOT_PATH / setting_name / MODELS_DIRECTORY_NAME

def _get_saved_model_path(setting_name, model_type, epoch_number):
	return _get_model_files_directory(setting_name) / get_model_filename(model_type, epoch_number)

@dataclass
class Statistics:
	training_loss: list[float]
	validation_loss: list[float]
	generator_loss: list[float]
	discriminator_loss: list[float]
	epoch_durations: list[float]

def _parse_float(value):
	return float(value.strip())

def _load_statistics(csv_path):
	training_loss = []
	validation_loss = []
	generator_losses = []
	discriminator_losses = []
	epoch_durations = []

	with csv_path.open(newline = "", encoding = "utf-8") as csv_file:
		reader = csv.DictReader(csv_file)
		for row in reader:
			for key, raw_value in row.items():
				value = _parse_float(raw_value)

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

def _plot_loss_series(all_colors, series, output_path, title):
	output_path.parent.mkdir(parents = True, exist_ok = True)
	plt.figure(figsize = (12, 6))

	for label, values in series:
		epochs = range(1, len(values) + 1)
		plt.plot(epochs, values, label = label, color = all_colors[label])

	plt.title(title)
	plt.xlabel(X_LABEL_NAMES["loss"])
	plt.ylabel(Y_LABEL_NAMES["loss"])
	if len(series) > 1:
		plt.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(output_path, format = "jpg")
	plt.close()

def _plot_combined_losses(color_list, stats_by_model):
	generator_series = []
	discriminator_series = []

	for model_name, stats in stats_by_model.items():  ##TODO refactor the handling of statistics, not good
		if stats.generator_loss:
			generator_series.append((model_name, stats.generator_loss))
		if stats.discriminator_loss:
			discriminator_series.append((model_name, stats.discriminator_loss))

	# original
	_plot_loss_series(color_list, generator_series, PATH_LOSS_PLOTS_PATH / f"{PLOT_IMAGE_NAMES['every_generator_loss']}.jpg", PLOT_NAMES['every_generator_loss'])
	_plot_loss_series(color_list, discriminator_series, PATH_LOSS_PLOTS_PATH / f"{PLOT_IMAGE_NAMES['every_discriminator_loss']}.jpg", PLOT_NAMES['every_discriminator_loss'])

	# by model_sizes
	plot_split_by_model_size(generator_series, discriminator_series)

	# by ls_size
	plot_split_by_ls_size(generator_series, discriminator_series)

def _plot_split_series(series, split_name, output_path, image_name_template, title_template):
	colors = get_colors_associated(generate_colors(len(series)), [name for name, _values in series])
	_plot_loss_series(
		colors,
		series,
		output_path / (image_name_template.replace("MODEL_NAME", split_name) + ".jpg"),
		title_template.replace("MODEL_NAME", split_name),
	)

def plot_split_by_generic(generator_series, discriminator_series, split_names, output_path, series_matches_split):
	for current_split_name in split_names:
		print('===> Current plot generator', current_split_name)
		_split_generator_series = [series for series in generator_series if series_matches_split(series[0], current_split_name)]
		_plot_split_series(_split_generator_series, current_split_name, output_path, PLOT_IMAGE_NAMES["split_generator_loss"], PLOT_NAMES["split_generator_loss"])

		print('===> Current plot discriminator', current_split_name)
		_split_discriminator_series = [series for series in discriminator_series if series_matches_split(series[0], current_split_name)]
		_plot_split_series(_split_discriminator_series, current_split_name, output_path, PLOT_IMAGE_NAMES["split_discriminator_loss"], PLOT_NAMES["split_discriminator_loss"])

def plot_split_by_model_size(generator_series, discriminator_series):
	plot_split_by_generic(
		generator_series,
		discriminator_series,
		ALL_MODELS,
		PATH_LOSS_PLOTS_BY_MODEL_PATH,
		lambda series_name, current_plot_model: series_name.startswith(current_plot_model),
	)

def plot_split_by_ls_size(generator_series, discriminator_series):
	ls_sizes_as_string = [get_ls_name(curr_ls) for curr_ls in LATENT_DIMENSION_GENERATOR_AVAILABLE]
	plot_split_by_generic(
		generator_series,
		discriminator_series,
		ls_sizes_as_string,
		PATH_LOSS_PLOTS_BY_LS_PATH,
		lambda series_name, current_plot_ls_size: series_name.endswith(current_plot_ls_size),
	)

def get_number_parameters(model_name, model_type):
	model_path = _get_model_files_directory(model_name)
	complete_models_list = sorted([path for path in model_path.iterdir() if path.is_file()])

	if model_type == DISCRIMINATOR_GLOBAL_NAME:
		total_path = complete_models_list[0]
	elif model_type == GENERATOR_GLOBAL_NAME:
		total_path = complete_models_list[-1]

	model = load_model(total_path)
	nb_params = sum([layer.count_params() for layer in model.layers if layer.trainable])
	return nb_params

def _parse_setting_name(setting_name):
	model_size, latent_space_name = setting_name.rsplit("-", 1)

	latent_space_size = int(latent_space_name.removeprefix(LATENT_SPACE_GLOBAL_NAME + "_"))

	return model_size, latent_space_size

def _is_configured_setting(setting_name):
	parsed_setting = _parse_setting_name(setting_name)
	if parsed_setting is None:
		return False

	model_size, latent_space_size = parsed_setting
	return model_size in ALL_MODELS and latent_space_size in LATENT_DIMENSION_GENERATOR_AVAILABLE

def _get_model_indexes(model_name):
	parsed_setting = _parse_setting_name(model_name)
	if parsed_setting is None:
		raise ValueError(f"Invalid model setting name: {model_name}")

	model_size, latent_size = parsed_setting
	idx_x = ALL_MODELS.index(model_size)
	idx_y = LATENT_DIMENSION_GENERATOR_AVAILABLE.index(latent_size)

	return idx_x, idx_y

def _get_contrasting_text_color(background_color):
	red, green, blue, _alpha = background_color
	background_intensity = (red + green + blue) / 3

	return 'black' if background_intensity > 0.5 else 'white'

def produce_heatmap(stats_by_model, output_dir, title, output_filename, value_getter, *, text_formatter):
	output_dir.mkdir(parents = True, exist_ok = True)
	data = np.zeros((len(ALL_MODELS), len(LATENT_DIMENSION_GENERATOR_AVAILABLE)))

	for model_name, stats in stats_by_model.items():
		idx_x, idx_y = _get_model_indexes(model_name)
		data[idx_x, idx_y] = value_getter(model_name, stats)

	x_labels = LATENT_DIMENSION_GENERATOR_AVAILABLE
	y_labels = ALL_MODELS

	plt.figure(figsize = (6, 5))

	heatmap = plt.imshow(data, cmap = 'grey')

	plt.title(title)
	plt.xlabel(X_LABEL_NAMES["heatmap"])
	plt.ylabel(Y_LABEL_NAMES["heatmap"])

	plt.xticks(ticks = np.arange(len(x_labels)), labels = x_labels)
	plt.yticks(ticks = np.arange(len(y_labels)), labels = y_labels)

	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			background_color = heatmap.cmap(heatmap.norm(data[i, j]))
			color = _get_contrasting_text_color(background_color)
			plt.text(j, i, text_formatter(data[i, j]), ha = 'center', va = 'center', color = color)

	plt.colorbar(heatmap)

	plt.savefig(output_dir / output_filename, format = 'jpg', dpi = 300)

	plt.show()

def _plot_current_number_epoch(stats_by_model, output_dir):
	produce_heatmap(
		stats_by_model,
		output_dir,
		PLOT_NAMES["current_number_epoch"],
		PLOT_IMAGE_NAMES["current_number_epoch"] + ".jpg",
		lambda _model_name, stats: len(stats.epoch_durations),
		text_formatter = lambda value: str(int(value)),
	)

def _plot_number_parameters(stats_by_model, output_dir, model_type):
	produce_heatmap(
		stats_by_model,
		output_dir,
		PLOT_NAMES["number_parameters"].replace("MODEL_NAME", model_type),
		PLOT_IMAGE_NAMES["number_parameters"].replace("MODEL_NAME", model_type) + ".jpg",
		lambda model_name, _stats: int(get_number_parameters(model_name, model_type)),
		text_formatter = lambda value: f"{int(value):,d}".replace(",", " "),
	)

def _plot_median_time_per_epoch(stats_by_model, output_dir):  # todo merge with time taken
	produce_heatmap(
		stats_by_model,
		output_dir,
		PLOT_NAMES["median_time_per_epoch"],
		PLOT_IMAGE_NAMES["median_time_per_epoch"] + ".jpg",
		lambda _model_name, stats: statistics.median(stats.epoch_durations),
		text_formatter = lambda value: str(round(value, 2)),
	)

def _collect_statistics_by_model():
	stats_by_model = {}

	if not MODELS_ROOT_PATH.is_dir():
		return stats_by_model

	for directory_path in sorted(MODELS_ROOT_PATH.iterdir()):
		csv_path = directory_path / STATISTICS_CSV_FILENAME
		stats = _load_statistics(csv_path)
		stats_by_model[directory_path.name] = stats

	return stats_by_model

def get_colors_associated(colors_list, stats):
	result = {}
	current_index = 0

	for name in stats:
		result[name] = colors_list[current_index]
		current_index += 1

	return result

def _ensure_plot_directories():
	for directory in (
			PLOTS_ROOT_DIRECTORY_PATH,
			PLOTS_NUMBER_PARAMETERS_PATH,
			PLOTS_HEATMAP_EPOCHS_PATH,
			PLOTS_HEATMAP_MODEL_SIZE_PATH,
			PLOTS_HEATMAP_LATENT_SPACE_SIZE_PATH,
			PATH_LOSS_PLOTS_PATH,
			PATH_LOSS_PLOTS_BY_LS_PATH,
			PATH_LOSS_PLOTS_BY_MODEL_PATH,
	):
		directory.mkdir(parents = True, exist_ok = True)

def _generate_combined_statistics_plots():
	stats_by_model = _collect_statistics_by_model()

	_ensure_plot_directories()

	colors_list_with_names = get_colors_associated(generate_colors(len(stats_by_model)), [name for name in stats_by_model.keys()])

	save_all_comparisons_models()

	_plot_combined_losses(colors_list_with_names, stats_by_model)

	_plot_current_number_epoch(stats_by_model, PLOTS_ROOT_DIRECTORY_PATH)

	_plot_number_parameters(stats_by_model, PLOTS_NUMBER_PARAMETERS_PATH, DISCRIMINATOR_GLOBAL_NAME)
	_plot_number_parameters(stats_by_model, PLOTS_NUMBER_PARAMETERS_PATH, GENERATOR_GLOBAL_NAME)

	_plot_median_time_per_epoch(stats_by_model, PLOTS_ROOT_DIRECTORY_PATH)

def get_real_images_sample():
	print('getting real images ')
	dataset_directory = DATASET_PATH

	image_paths = [path for path in sorted(dataset_directory.iterdir()) if path.is_file()]

	selected_paths = random.sample(image_paths, k = min(NUMBER_COMPARISON, len(image_paths)))
	images = []

	for image_path in selected_paths:
		normalized_image = _load_image(image_path)

		images.append(img_to_array(normalized_image))

	return images

def get_fake_images_sample(generator_name, generator_epoch):
	print('Generating fake images using ', generator_name, generator_epoch)
	epoch_number = int(str(generator_epoch).replace(str(EPOCH_GLOBAL_NAME + "_"), ""))

	generator_path = _get_saved_model_path(generator_name, GENERATOR_GLOBAL_NAME, epoch_number)

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
	# could use get_model_path_at_given_epoch(model_type, current_best_result) ? i don't know
	# should obtain the accuracy of a given model on a given epoch, using a special set of images. they are either all real or all fake, given by boolean is_real_images

	epoch_number = int(str(model_epoch).replace(str(EPOCH_GLOBAL_NAME + "_"), ""))

	model_path = _get_saved_model_path(model_name, DISCRIMINATOR_GLOBAL_NAME, epoch_number)

	discriminator = keras.models.load_model(model_path)
	images_array = np.asarray(images_set, dtype = np.float32)
	predictions = np.squeeze(discriminator(images_array, training = False).numpy())

	expected_label = 1.0 if is_real_images else 0.0
	predicted_labels = (predictions >= 0.5).astype(np.float32)

	return float(np.mean(predicted_labels == expected_label))

def get_values_comparisons(size, comparisons_elements):
	result = [[0 for _ in range(size)] for _ in range(size + 1)]
	# nb_comparisons

	for current_generator_index in range(size + 1):
		################################################################################# First, generate images
		if current_generator_index == 0:
			real_images = True
			generated_images = get_real_images_sample()
		else:
			real_images = False
			generated_images = get_fake_images_sample(comparisons_elements[current_generator_index - 1][0], comparisons_elements[current_generator_index - 1][1])

		################################################################################# then, see what discriminators could have discriminated them

		for current_discriminator_index in range(size):
			model_name = comparisons_elements[current_discriminator_index][0]
			epoch = comparisons_elements[current_discriminator_index][1]
			accuracy = get_accuracy_on_images(model_name, epoch, generated_images, real_images)
			result[current_generator_index][int(current_discriminator_index)] = accuracy

	return result

def save_all_comparisons_models():
	print('\n======> Generate heatmap epoch')
	produce_heatmap_epoch()

	print('\n======> Generate heatmap model size')
	produce_heatmap_model_size()

	print('\n======> Generate heatmap latent space')
	produce_heatmap_latent_space()

def get_number_epoch_in_given_setting(setting):
	setting_models_directory = _get_model_files_directory(setting)

	max_epoch = 0

	for model_file in setting_models_directory.iterdir():
		model_details = parse_model_filename(model_file.name)
		current_epoch, _model_type = model_details
		max_epoch = max(max_epoch, current_epoch)

	return max_epoch

def produce_heatmap_epoch():
	available_settings = [entry.name for entry in MODELS_ROOT_PATH.iterdir() if entry.is_dir()]
	if PLOTS_ROOT_DIRECTORY_PATH.name in available_settings:
		available_settings.remove(PLOTS_ROOT_DIRECTORY_PATH.name)

	for current_setting in available_settings:
		print("====> Current setting : ", current_setting)
		max_epoch = get_number_epoch_in_given_setting(current_setting)
		if max_epoch == 100:
			print("==> Has 100 epoch")
			step = int(max_epoch / NUMBER_EPOCH_TAKEN_COMPARISON)
			current_epoch = 0
			comparisons_elements = []
			for i in range(NUMBER_EPOCH_TAKEN_COMPARISON + 1):
				epoch_name = get_epoch_name(current_epoch)
				comparisons_elements.append((current_setting, epoch_name))
				current_epoch = current_epoch + step

			save_comparisons_models(
				comparisons_elements,
				PLOTS_HEATMAP_EPOCHS_PATH,
				PLOT_NAMES["comparison_heatmap"].replace("MODEL_NAME", current_setting),
				PLOT_IMAGE_NAMES["comparison_heatmap"].replace("MODEL_NAME", current_setting) + ".png",
			)

def get_epoch_name(current_epoch):  # TODO use this in train etc
	return EPOCH_GLOBAL_NAME + "_" + ((6 - len(str(current_epoch))) * "0") + str(current_epoch)

def get_ls_name(current_latent_dimension_generator):
	return LATENT_SPACE_GLOBAL_NAME + "_" + ((4 - len(str(current_latent_dimension_generator))) * "0") + str(current_latent_dimension_generator)

def produce_heatmap_model_size():
	for current_latent_dimension_generator in LATENT_DIMENSION_GENERATOR_AVAILABLE:
		comparisons_elements = []  # list every model size for that ls
		current_latent_dimension_generator_str = get_ls_name(current_latent_dimension_generator)

		print("====> now on ", current_latent_dimension_generator_str)

		for current_model in ALL_MODELS:
			total_name = current_model + "-" + current_latent_dimension_generator_str
			available_epochs = get_number_epoch_in_given_setting(total_name)
			epoch_name = get_epoch_name(available_epochs)

			new_elem = (total_name, epoch_name)
			comparisons_elements.append(new_elem)

			print("==> Current model : ", current_model, " nb epochs ", epoch_name, " result ", new_elem)

		save_comparisons_models(
			comparisons_elements,
			PLOTS_HEATMAP_MODEL_SIZE_PATH,
			PLOT_NAMES["latent_space_size_comparison_heatmap"].replace("LATENT_SPACE_SIZE", str(current_latent_dimension_generator)),
			PLOT_IMAGE_NAMES["latent_space_size_comparison_heatmap"].replace("LATENT_SPACE_SIZE", str(current_latent_dimension_generator)) + ".png",
		)

def produce_heatmap_latent_space():
	for current_model in ALL_MODELS:
		comparisons_elements = []  # list every ls for that model size
		print("====> now on ", current_model)
		for current_ls in LATENT_DIMENSION_GENERATOR_AVAILABLE:
			current_latent_dimension_generator_str = get_ls_name(current_ls)
			print("==> Current ls ", current_latent_dimension_generator_str)
			total_name = current_model + "-" + current_latent_dimension_generator_str
			epoch_name = get_epoch_name(get_number_epoch_in_given_setting(total_name))
			new_elem = (total_name, epoch_name)
			comparisons_elements.append(new_elem)

		save_comparisons_models(
			comparisons_elements,
			PLOTS_HEATMAP_LATENT_SPACE_SIZE_PATH,
			PLOT_NAMES["model_size_comparison_heatmap"].replace("MODEL_NAME", current_model),
			PLOT_IMAGE_NAMES["model_size_comparison_heatmap"].replace("MODEL_NAME", current_model) + ".png",
		)

def generate_colors(n):
	colors = []
	for i in range(n):
		r, g, b = colorsys.hsv_to_rgb(i / n, 1.0, 1.0)

		colors.append("#{0:02x}{1:02x}{2:02x}".format(int(r * 255), int(g * 255), int(b * 255)))

	return colors

def save_comparisons_models(comparisons_elements, directory, title, output_filename):
	directory.mkdir(parents = True, exist_ok = True)
	size = len(comparisons_elements)

	data = get_values_comparisons(size, comparisons_elements)

	# todo detect if only one element differs
	# row_labels = ["real images"] + [elem[0] + "\n" + elem[1] for elem in comparisons_elements]
	# col_labels = [elem[0] + "\n" + elem[1] for elem in comparisons_elements]
	row_labels = ["real images"] + [elem[0] + elem[1] for elem in comparisons_elements]
	col_labels = [elem[0] + elem[1] for elem in comparisons_elements]

	fig, ax = plt.subplots()
	im = ax.imshow(data, cmap = 'gray', interpolation = 'nearest', vmin = 0, vmax = 1)

	ax.set_xticks(np.arange(len(col_labels)))
	ax.set_yticks(np.arange(len(row_labels)))
	ax.set_xticklabels(col_labels)
	ax.set_yticklabels(row_labels)

	# plt.setp(ax.get_xticklabels(), rotation = 75, ha = "right", rotation_mode = "anchor")
	plt.setp(ax.get_xticklabels(), rotation = 90, ha = "right", rotation_mode = "anchor")

	plt.colorbar(im)
	plt.title(title)
	ax.set_xlabel(X_LABEL_NAMES["comparison_heatmap"])
	ax.set_ylabel(Y_LABEL_NAMES["comparison_heatmap"])

	for i in range(len(data)):
		for j in range(len(data[0])):
			data_as_percentage = data[i][j] * 100
			if data_as_percentage < 50:
				color = "white"
			else:
				color = "black"

			# ax.text(j, i, str(round(data_as_percentage, 1)) + "%", ha = "center", va = "center", color = color, fontsize = 8)
			ax.text(j, i, str(round(data_as_percentage, 1)) + "%", ha = "center", va = "center", color = color, fontsize = 4)

	plt.subplots_adjust(bottom = 0.6)
	# plt.savefig(Path(PLOTS_ROOT_DIRECTORY_PATH, "heatmap.png"), dpi = 300)
	plt.savefig(Path(directory) / output_filename, dpi = 300)
	plt.close()

def _get_available_generator_paths():
	models_dir = Path(MODELS_DIRECTORY)
	generator_paths = []
	for model_name in get_list_of_keras_models(str(models_dir)):
		model_details = parse_model_filename(model_name)

		_epoch, model_type = model_details
		if model_type == GENERATOR_GLOBAL_NAME:
			generator_paths.append(models_dir / model_name)

	return generator_paths

def _prepare_output_image(image):
	if not IS_RGB_IMAGES and image.shape[-1] == 3:
		image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		image = np.expand_dims(image, axis = -1)

	output_image = _denormalize_images(img_to_array(image.astype("float32")))

	if output_image.shape[-1] == 3:
		output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

	return output_image

def save_evolution_sample_images(generator_name):
	generator_paths = _get_available_generator_paths()
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
		output_image = _prepare_output_image(image)

		epoch, _model_type = parse_model_filename(generator_path.name)
		output_path = output_dir / f"{EVOLUTION_SAMPLE_PREFIX}_{epoch:06d}.png"
		cv2.imwrite(str(output_path), output_image)

def produce_evolution_sample():
	generator_name = f"{MODEL_NAME}-ls_{LATENT_DIMENSION_GENERATOR:04d}"
	save_evolution_sample_images(generator_name)

def apply_model(generator, latent_vector):
	latent_array = np.expand_dims(np.asarray(latent_vector, dtype = "float32"), axis = 0)
	image = generator(latent_array, training = False).numpy()[0]
	return _prepare_output_image(image)

def get_difference_with_original(generator, latent_vector, goal):
	reproduced_image = apply_model(generator, latent_vector)
	# print("shape : ", goal.shape, reproduced_image.shape, type(goal), type(reproduced_image))
	delta = np.abs(goal - reproduced_image)
	return delta.reshape(-1).sum()

def _random_latent_value():
	return round(random.gauss(0, 1), 2)

def _random_latent_vector(latent_space_size):
	return [_random_latent_value() for _ in range(latent_space_size)]

def _is_better_candidate(candidate_difference, best_difference):
	return best_difference is None or candidate_difference < best_difference

def search_random(generator, goal, ls_size, quantity_initial_random):  # returns best latent vector
	best_latent_vector = None
	best_difference = None

	for generation_index in range(quantity_initial_random):
		if (generation_index % 100) == 0:
			print("===> Search random, ", generation_index, "/", quantity_initial_random)

		candidate_vector = _random_latent_vector(ls_size)
		candidate_difference = get_difference_with_original(generator, candidate_vector, goal)
		if _is_better_candidate(candidate_difference, best_difference):
			print("==> new best difference : ", candidate_difference)
			best_latent_vector = candidate_vector.copy()
			best_difference = candidate_difference

	return best_latent_vector

def get_rnd_elem():
	return _random_latent_value()

def mutate_vector(current_vector, nb_diff):
	new_vector = current_vector.copy()

	for _ in range(nb_diff):
		mutation_index = random.randint(0, len(current_vector) - 1)
		new_vector[mutation_index] = _random_latent_value()

	return new_vector

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

def save_produced_result(generator, latent_vector, output_path):
	result = apply_model(generator, latent_vector)
	output_path.parent.mkdir(parents = True, exist_ok = True)

	cv2.imwrite(str(output_path), result)

def reproduction_search():
	# open generator
	models_dir = Path(MODELS_DIRECTORY)
	gen_epoch = get_last_epoch_available(GENERATOR_GLOBAL_NAME, str(models_dir))
	generator_path = models_dir / get_model_filename(GENERATOR_GLOBAL_NAME, gen_epoch)
	output_dir = Path(REPRODUCED_IMAGES_OUTPUT_DIRECTORY)
	generator = keras.models.load_model(generator_path)

	# open goal image
	goal = keras.utils.img_to_array(keras.utils.load_img(IMAGE_TO_REPRODUCE))

	all_best_latent_vectors = []

	for current_avg in range(NB_RETRIES_AVG):
		print("========> NEW ,", current_avg)
		best_latent_vector = search_random(generator, goal, LATENT_DIMENSION_GENERATOR, QUANTITY_INITIAL_RANDOM)

		best_latent_vector = search_genetic_algorithm(generator, best_latent_vector, goal, QUANTITY_GENETIC_EVO, QUANTITY_GENETIC_ALGO)

		# produce best image and save it
		save_produced_result(generator, best_latent_vector, output_dir / str("tmp_" + REPRODUCED_IMAGE_SUFFIX + str(current_avg) + ".png"))

		all_best_latent_vectors.append(best_latent_vector)
		print('==> finished, this Result : ', best_latent_vector)

	overall_avg_latent_vector = get_avg_latent_vector(all_best_latent_vectors)

	save_produced_result(generator, overall_avg_latent_vector, output_dir / str(REPRODUCED_IMAGE_SUFFIX + ".png"))

	print('==> Result : ', overall_avg_latent_vector)
	print("==> Total diff : ", get_difference_with_original(generator, overall_avg_latent_vector, goal))

def get_avg_latent_vector(all_best_latent_vectors):
	return np.mean(np.asarray(all_best_latent_vectors, dtype = np.float32), axis = 0).tolist()

def produce_continuous_movement():
	generator_name = f"{MODEL_NAME}-ls_{LATENT_DIMENSION_GENERATOR:04d}"
	generate_fake_images_sample(generator_name, CONTINUOUS_MOVEMENT_LENGTH, CONTINUOUS_MOVEMENT_NUMBER_CHANGES)

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

		_mutate_latent_vector_in_place(latent_vector, nb_changes)

		image = generator(latent_vector, training = False).numpy()[0]
		output_image = _prepare_output_image(image)

		output_path = output_dir / str(CONTINUOUS_MOVEMENT_IMAGE_PREFIX + f"_{image_index + 1:04d}.png")
		cv2.imwrite(str(output_path), output_image)

def _mutate_latent_vector_in_place(latent_vector, nb_changes):
	indices = np.random.choice(latent_vector.shape[1], size = nb_changes, replace = False)
	new_values = np.random.normal(0.0, 1.0, size = nb_changes)
	latent_vector[0, indices] = new_values

def _load_model(model_path):
	return tf.keras.models.load_model(str(model_path), compile = False)

def _build_concrete_function(model):
	input_specs = [tf.TensorSpec(shape = [dim if dim is not None else 1 for dim in tensor.shape], dtype = tensor.dtype) for tensor in model.inputs]

	@tf.function
	def model_fn(*args):
		return model(*args, training = False)

	return model_fn.get_concrete_function(*input_specs)

def _configure_converter(concrete_fn, model):
	converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], model)
	converter.experimental_new_converter = False
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
	converter.allow_custom_ops = False
	converter.optimizations = []
	return converter

def export_tflite(model_path_keras, model_path_tflite):
	model = _load_model(model_path_keras)
	concrete_function = _build_concrete_function(model)
	converter = _configure_converter(concrete_function, model)
	tflite_model = converter.convert()

	target_path = Path(model_path_tflite)
	target_path.parent.mkdir(parents = True, exist_ok = True)
	target_path.write_bytes(tflite_model)

def _default_models():
	last_generator = None
	last_discriminator = None

	for model_name in get_list_of_keras_models():
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
	for source, target in _default_models():
		print(f"Converting {source} -> {target}")
		export_tflite(source, target)

def get_generator_model_path_at_given_epoch(epoch):
	return get_model_path_at_given_epoch("generator", epoch)

def get_discriminator_model_path_at_given_epoch(epoch):
	return get_model_path_at_given_epoch("discriminator", epoch)

def _model_directory_for(model_name, latent_space_size):
	return os.path.join(
		MODELS_ROOT_PATH,
		f"{model_name}-ls_{latent_space_size:04d}",
		MODELS_DIRECTORY_NAME,
	)

def get_model_path_at_given_epoch(model_type, epoch, models_dir = MODELS_DIRECTORY):
	return os.path.join(models_dir, get_model_filename(model_type, epoch))

def get_model_path_at_given_epoch_closest_possible(model_type, epoch, available_epochs, models_dir = MODELS_DIRECTORY):
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

def get_available_epochs(models_dir = MODELS_DIRECTORY):
	models_list = get_list_of_keras_models(models_dir)
	return [epoch for epoch, model_type in filter(None, (parse_model_filename(model) for model in models_list)) if model_type == DISCRIMINATOR_GLOBAL_NAME]

def _indexes_to_load(models_quantity):
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
	models_dir = _model_directory_for(model_name, latent_space_size)

	models_quantity = get_current_epoch(models_dir)
	indexes = _indexes_to_load(models_quantity)

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

def get_list_of_keras_models(models_dir = MODELS_DIRECTORY):
	if not os.path.isdir(models_dir):
		return []

	complete_list = sorted(os.listdir(models_dir))
	return [filename for filename in complete_list]  # if not filename.endswith(".csv")

def get_current_epoch(models_dir = MODELS_DIRECTORY):
	keras_models = get_list_of_keras_models(models_dir)
	available_epochs = [epoch for epoch, _model_type in filter(None, (parse_model_filename(model) for model in keras_models))]
	return max(available_epochs)

def get_last_epoch_available(model_type, models_dir = MODELS_DIRECTORY):
	models_list = get_list_of_keras_models(models_dir)
	candidates = [
		epoch
		for epoch, current_model_type in filter(None, (parse_model_filename(model) for model in models_list))
		if current_model_type == model_type
	]

	if not candidates:
		raise ValueError(f"No {model_type} models available in {models_dir}.")

	return max(candidates)
