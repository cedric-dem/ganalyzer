from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import colorsys
import numpy as np
import random
import cv2
import keras
from keras.preprocessing.image import img_to_array
import statistics
from tensorflow.keras.models import load_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PLOTS_ROOT_DIRECTORY, RESULTS_ROOT_PATH, IS_RGB_IMAGES, NUMBER_COMPARISON, DATASET_PATH, LATENT_DIMENSION_GENERATOR_AVAILABLE, MODELS_ROOT_PATH, NUMBER_EPOCH_TAKEN_COMPARISON, PLOTS_HEATMAP_EPOCHS_DIRECTORY, \
	PLOTS_HEATMAP_MODEL_SIZE_DIRECTORY, PLOTS_HEATMAP_LATENT_SPACE_SIZE_DIRECTORY, PATH_LOSS_PLOTS, PATH_LOSS_BY_LS_PLOTS, PATH_LOSS_BY_MODEL_PLOTS, PLOTS_NUMBER_PARAMETERS_DIRECTORY, ALL_MODELS, DISCRIMINATOR_GLOBAL_NAME, GENERATOR_GLOBAL_NAME, EPOCH_GLOBAL_NAME, MODELS_DIRECTORY_NAME, \
	PLOT_IMAGE_NAMES, PLOT_NAMES, LATENT_SPACE_GLOBAL_NAME, X_LABEL_NAMES, Y_LABEL_NAMES, STATISTICS_CSV_FILENAME

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

def _get_model_files_directory(setting_name):
	return MODELS_ROOT_PATH / setting_name / MODELS_DIRECTORY_NAME

def _get_saved_model_path(setting_name, model_type, epoch_number):
	return _get_model_files_directory(setting_name) / f"{model_type}_{EPOCH_GLOBAL_NAME}_{epoch_number:06d}.keras"

@dataclass
class Statistics:
	training_loss: list[float]
	validation_loss: list[float]
	generator_loss: list[float]
	discriminator_loss: list[float]
	epoch_durations: list[float]

def _parse_float(value):
	if value is None:
		return None

	stripped_value = value.strip()
	if not stripped_value:
		return None

	try:
		return float(stripped_value)
	except ValueError:
		return None

def _load_statistics(csv_path):
	training_loss = []
	validation_loss = []
	generator_loss = []
	discriminator_loss = []
	epoch_durations = []

	with csv_path.open(newline = "", encoding = "utf-8") as csv_file:
		reader = csv.DictReader(csv_file)
		for row in reader:
			for key, raw_value in row.items():
				value = _parse_float(raw_value)
				if value is None:
					continue

				lower_key = key.lower()
				if "gen" in lower_key and "loss" in lower_key:
					generator_loss.append(value)
				elif "disc" in lower_key and "loss" in lower_key:
					discriminator_loss.append(value)
				elif "val_loss" in lower_key or lower_key == "validation_loss":
					validation_loss.append(value)
				elif lower_key == "loss" or lower_key.endswith("_loss"):
					training_loss.append(value)
				elif "time" in lower_key or "duration" in lower_key:
					epoch_durations.append(value)

	return Statistics(
		training_loss = training_loss,
		validation_loss = validation_loss,
		generator_loss = generator_loss,
		discriminator_loss = discriminator_loss,
		epoch_durations = epoch_durations,
	)

def _plot_loss_series(all_colors, series, output_path, title):
	output_path.parent.mkdir(parents = True, exist_ok = True)
	plt.figure(figsize = (12, 6))

	current_idx = 0
	for label, values in series:
		if not values:
			continue

		epochs = range(1, len(values) + 1)
		plt.plot(epochs, values, label = label, color = all_colors[label])

		current_idx += 1

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

def plot_split_by_generic(generator_series, discriminator_series, split_names, output_path, series_matches_split):
	for current_split_name in split_names:
		print('===> Current plot generator', current_split_name)
		this_generator_series = [
			current_elem_in_series
			for current_elem_in_series in generator_series
			if series_matches_split(current_elem_in_series[0], current_split_name)
		]

		color_list = get_colors_associated(generate_colors(len(this_generator_series)), [name[0] for name in this_generator_series])
		_plot_loss_series(color_list, this_generator_series, output_path / (PLOT_IMAGE_NAMES["split_generator_loss"].replace("MODEL_NAME", current_split_name) + ".jpg"), PLOT_NAMES["split_generator_loss"].replace("MODEL_NAME", current_split_name))

		print('===> Current plot discriminator', current_split_name)
		this_discriminator_series = [
			current_elem_in_series
			for current_elem_in_series in discriminator_series
			if series_matches_split(current_elem_in_series[0], current_split_name)
		]

		color_list = get_colors_associated(generate_colors(len(this_discriminator_series)), [name[0] for name in this_discriminator_series])
		_plot_loss_series(color_list, this_discriminator_series, output_path / (PLOT_IMAGE_NAMES["split_discriminator_loss"].replace("MODEL_NAME", current_split_name) + ".jpg"), PLOT_NAMES["split_discriminator_loss"].replace("MODEL_NAME", current_split_name))

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

	if not complete_models_list:
		return 0

	if model_type == DISCRIMINATOR_GLOBAL_NAME:
		total_path = complete_models_list[0]
	elif model_type == GENERATOR_GLOBAL_NAME:
		total_path = complete_models_list[-1]
	else:
		return 0

	model = load_model(total_path)
	nb_params = sum([layer.count_params() for layer in model.layers if layer.trainable])
	return nb_params

def _parse_setting_name(setting_name):
	if "-" not in setting_name:
		return None

	model_size, latent_space_name = setting_name.rsplit("-", 1)
	if not latent_space_name.startswith(LATENT_SPACE_GLOBAL_NAME + "_"):
		return None

	try:
		latent_space_size = int(latent_space_name.removeprefix(LATENT_SPACE_GLOBAL_NAME + "_"))
	except ValueError:
		return None

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
		if not directory_path.is_dir():
			continue

		if not _is_configured_setting(directory_path.name):
			continue

		csv_path = directory_path / STATISTICS_CSV_FILENAME
		if not csv_path.exists():
			continue

		stats = _load_statistics(csv_path)
		if not (stats.generator_loss or stats.discriminator_loss or stats.epoch_durations):
			continue

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
	read_mode = cv2.IMREAD_COLOR if IS_RGB_IMAGES else cv2.IMREAD_GRAYSCALE

	for image_path in selected_paths:
		image = cv2.imread(str(image_path), read_mode)
		if image is None:
			print(f"Failed to load image: {image_path}")
			continue

		if IS_RGB_IMAGES:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		normalized = (image.astype("float32") - 127.5) / 127.5
		images.append(img_to_array(normalized))

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

	if not images_set:
		return 0.0

	epoch_number = int(str(model_epoch).replace(str(EPOCH_GLOBAL_NAME + "_"), ""))

	model_path = _get_saved_model_path(model_name, DISCRIMINATOR_GLOBAL_NAME, epoch_number)

	discriminator = keras.models.load_model(model_path)
	images_array = np.asarray(images_set, dtype = np.float32)
	predictions = np.squeeze(discriminator(images_array, training = False).numpy())

	expected_label = 1.0 if is_real_images else 0.0
	predicted_labels = (predictions >= 0.5).astype(np.float32)

	return float(np.mean(predicted_labels == expected_label))

def get_values_comparisons(size, comparisons_elements):
	result = [[0 for _ in range(size)] for j in range(size + 1)]
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
			result[current_generator_index][current_discriminator_index] = accuracy

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

	if not setting_models_directory.exists():
		return 0

	max_epoch = 0

	for model_file in setting_models_directory.iterdir():
		if not model_file.is_file():
			continue

		if not model_file.name.endswith(".keras"):
			continue

		try:
			current_epoch = int(model_file.stem.split("_")[-1])
		except ValueError:
			continue

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

if __name__ == "__main__":
	_generate_combined_statistics_plots()
