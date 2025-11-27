from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import random
import cv2
import keras
from keras.preprocessing.image import img_to_array

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PLOTS_ROOT_DIRECTORY, every_models_statistics_path, results_root_path, rgb_images, nb_comparisons, dataset_path, latent_dimension_generator, latent_dimension_generator_available, models_directory, nb_epoch_taken_comparison, PLOTS_HEATMAP_EPOCHS_DIRECTORY, \
	PLOTS_HEATMAP_MODEL_SIZE_DIRECTORY, PLOTS_HEATMAP_LATENT_SPACE_SIZE_DIRECTORY
from ganalyzer.model_config import all_models

STATISTICS_FILENAME = "statistics.csv"
PLOTS_ROOT_DIRECTORY_PATH = Path(PLOTS_ROOT_DIRECTORY)

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

def _plot_loss_series(l_settings, series, output_path, title, xlabel = "Epoch", ylabel = "Loss"):
	plt.figure()
	plotted_any = False

	for label, values in series:
		if not values:
			continue

		epochs = range(1, len(values) + 1)
		plt.plot(epochs, values, label = label, color = _color_for_label(label, l_settings))
		plotted_any = True

	if not plotted_any:
		plt.close()
		return False

	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	if len(series) > 1:
		plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(output_path, format = "jpg")
	plt.close()
	return True

def _color_for_label(label, lst_settings):
	# those two just count the position in the list of sorted models settings
	# proportion = lst_settings.index(label) / (len(lst_settings) - 2)
	# color_direction = [[0, 1], [1, -1], [0, 0]]

	# here it split between small, medium and large models
	proportion = [s for s in lst_settings if s.startswith(label[:7])].index(label) / 3
	if label.startswith("model_0"):
		color_direction = [[0, 1], [1, -1], [0, 0]]
	elif label.startswith("model_1"):
		color_direction = [[0, 0], [0, 1], [1, -1]]
	elif label.startswith("model_2"):
		color_direction = [[1, -1], [0, 0], [0, 1]]
	elif label.startswith("model_3"):
		color_direction = [[0, 0], [1, -1], [0, 1]]
	else:
		print("not found col dir")
	return _proportion_to_color(proportion, color_direction)

def _proportion_to_color(proportion, color_direction):
	clamped = max(0.0, min(1.0, proportion))

	red = int(255 * (color_direction[0][0] + (color_direction[0][1] * clamped)))
	green = int(255 * (color_direction[1][0] + (color_direction[1][1] * clamped)))
	blue = int(255 * (color_direction[2][0] + (color_direction[2][1] * clamped)))

	return f"#{red:02x}{green:02x}{blue:02x}"

def _plot_combined_losses(l_settings, stats_by_model, output_dir):
	generator_series = []
	discriminator_series = []

	for model_name, stats in stats_by_model.items():
		if stats.generator_loss:
			generator_series.append((model_name, stats.generator_loss))
		if stats.discriminator_loss:
			discriminator_series.append((model_name, stats.discriminator_loss))

	plotted_any = False
	generator_plot = output_dir / "generator_loss.jpg"
	discriminator_plot = output_dir / "discriminator_loss.jpg"

	if generator_series and _plot_loss_series(l_settings, generator_series, generator_plot, "Generator Loss Over Epochs"):
		plotted_any = True

	if discriminator_series and _plot_loss_series(l_settings, discriminator_series, discriminator_plot, "Discriminator Loss Over Epochs"):
		plotted_any = True

	return plotted_any

def _plot_combined_epoch_times(l_settings, stats_by_model, output_dir):
	plt.figure()
	plotted_any = False

	for model_name, stats in stats_by_model.items():
		if not stats.epoch_durations:
			continue

		epochs = range(1, len(stats.epoch_durations) + 1)
		plt.plot(epochs, stats.epoch_durations, label = model_name, color = _color_for_label(model_name, l_settings))
		plotted_any = True

	if not plotted_any:
		plt.close()
		return False

	plt.title("Epoch Duration")
	plt.xlabel("Epoch")
	plt.ylabel("Time (seconds)")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(output_dir / "times_epoch.jpg", format = "jpg")
	plt.close()
	return True

def _collect_statistics_by_model():
	stats_by_model = {}

	for model_directory in every_models_statistics_path:
		directory_path = Path(model_directory)
		csv_path = directory_path / STATISTICS_FILENAME
		if not csv_path.exists():
			continue

		stats = _load_statistics(csv_path)
		if not (stats.generator_loss or stats.discriminator_loss or stats.epoch_durations):
			continue

		stats_by_model[directory_path.name] = stats

	return stats_by_model

def _generate_combined_statistics_plots():
	stats_by_model = _collect_statistics_by_model()
	if not stats_by_model:
		print("No statistics were found to generate combined plots.")
		return

	has_generator_loss = any(stats.generator_loss for stats in stats_by_model.values())
	has_discriminator_loss = any(stats.discriminator_loss for stats in stats_by_model.values())
	has_epoch_durations = any(stats.epoch_durations for stats in stats_by_model.values())

	if not (has_generator_loss or has_discriminator_loss or has_epoch_durations):
		print("No numeric statistics were found to plot for combined statistics.")
		return

	PLOTS_ROOT_DIRECTORY_PATH.mkdir(parents = True, exist_ok = True)

	l_settings = os.listdir(results_root_path)
	l_settings.sort()

	plotted_any = False
	if has_generator_loss or has_discriminator_loss:
		if _plot_combined_losses(l_settings, stats_by_model, PLOTS_ROOT_DIRECTORY_PATH):
			plotted_any = True

	if has_epoch_durations and _plot_combined_epoch_times(l_settings, stats_by_model, PLOTS_ROOT_DIRECTORY_PATH):
		plotted_any = True

	if plotted_any:
		print(f"Saved combined statistics plots to '{PLOTS_ROOT_DIRECTORY_PATH}'.")
	else:
		print("No numeric statistics were found to plot for combined statistics.")

	save_all_comparisons_models()

def get_real_images_sample():
	print('getting real images ')
	dataset_directory = Path(dataset_path)

	image_paths = [path for path in sorted(dataset_directory.iterdir()) if path.is_file()]

	selected_paths = random.sample(image_paths, k = min(nb_comparisons, len(image_paths)))
	images = []
	read_mode = cv2.IMREAD_COLOR if rgb_images else cv2.IMREAD_GRAYSCALE

	for image_path in selected_paths:
		image = cv2.imread(str(image_path), read_mode)
		if image is None:
			print(f"Failed to load image: {image_path}")
			continue

		if rgb_images:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		normalized = (image.astype("float32") - 127.5) / 127.5
		images.append(img_to_array(normalized))

	return images

def get_fake_images_sample(generator_name, generator_epoch):
	print('Generating fake images using ', generator_name, generator_epoch)
	epoch_number = int(str(generator_epoch).replace("epoch_", ""))

	generator_path = Path(results_root_path) / generator_name / "models" / f"generator_epoch_{epoch_number:06d}.keras"

	generator = keras.models.load_model(generator_path)
	ls_size = int(generator_name.split("_")[-1])
	latent_vectors = np.random.normal(0.0, 1.0, size = (nb_comparisons, ls_size))

	generated_images = generator(latent_vectors, training = False).numpy()
	images = []

	for image in generated_images:
		if not rgb_images and image.shape[-1] == 3:
			image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			image = np.expand_dims(image, axis = -1)

		images.append(img_to_array(image.astype("float32")))

	return images

def get_accuracy_on_images(model_name, model_epoch, images_set, is_real_images):
	# could use get_model_path_at_given_epoch(model_type, current_best_result) ? i don't know
	# should obtain the accuracy of a given model on a given epoch, using a special set of images. they are either all real or all fake, given by boolean is_real_images

	if not images_set:
		return 0.0

	epoch_number = int(str(model_epoch).replace("epoch_", ""))

	model_path = Path(results_root_path) / model_name / "models" / f"discriminator_epoch_{epoch_number:06d}.keras"

	discriminator = keras.models.load_model(model_path)
	images_array = np.asarray(images_set, dtype = np.float32)
	predictions = np.squeeze(discriminator(images_array, training = False).numpy())

	expected_label = 1.0 if is_real_images else 0.0
	predicted_labels = (predictions >= 0.5).astype(np.float32)

	return float(np.mean(predicted_labels == expected_label))

def get_values_comparisons(size, comparisons_elements):
	result = [[0 for i in range(size)] for j in range(size + 1)]
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
	setting_models_directory = Path(results_root_path) / setting / "models"

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

	print('========> debug ', setting, results_root_path, max_epoch)
	return max_epoch

def produce_heatmap_epoch():
	available_settings = os.listdir(results_root_path)
	if "plots" in available_settings:
		available_settings.remove("plots")

	for current_setting in available_settings:
		print("====> Current setting : ", current_setting)
		max_epoch = get_number_epoch_in_given_setting(current_setting)
		if max_epoch == 100:
			print("==> Has 100 epoch")
			step = int(max_epoch / nb_epoch_taken_comparison)
			current_epoch = 0
			comparisons_elements = []
			for i in range(nb_epoch_taken_comparison + 1):
				epoch_name = get_epoch_name(current_epoch)
				comparisons_elements.append((current_setting, epoch_name))
				current_epoch = current_epoch + step

			save_comparisons_models(comparisons_elements, PLOTS_HEATMAP_EPOCHS_DIRECTORY, current_setting)

def get_epoch_name(current_epoch):  # TODO use this in train etc
	return "epoch_" + ((6 - len(str(current_epoch))) * "0") + str(current_epoch)

def get_ls_name(current_latent_dimension_generator):
	return "ls_" + ((4 - len(str(current_latent_dimension_generator))) * "0") + str(current_latent_dimension_generator)

def produce_heatmap_model_size():
	for current_latent_dimension_generator in latent_dimension_generator_available:
		comparisons_elements = []  # list every model size for that ls
		current_latent_dimension_generator_str = get_ls_name(current_latent_dimension_generator)

		print("====> now on ", current_latent_dimension_generator_str)

		for current_model in all_models:
			total_name = current_model + "-" + current_latent_dimension_generator_str
			available_epochs = get_number_epoch_in_given_setting(total_name)
			epoch_name = get_epoch_name(available_epochs)

			new_elem = (total_name, epoch_name)
			comparisons_elements.append(new_elem)

			print("==> Current model : ", current_model, " nb epochs ", epoch_name, " result ", new_elem)

		save_comparisons_models(comparisons_elements, PLOTS_HEATMAP_MODEL_SIZE_DIRECTORY, "ls_size =  " + str(current_latent_dimension_generator))

def produce_heatmap_latent_space():
	for current_model in all_models:
		comparisons_elements = []  # list every ls for that model size
		print("====> now on ", current_model)
		for current_ls in latent_dimension_generator_available:
			current_latent_dimension_generator_str = get_ls_name(current_ls)
			print("==> Current ls ", current_latent_dimension_generator_str)
			total_name = current_model + "-" + current_latent_dimension_generator_str
			epoch_name = get_epoch_name(get_number_epoch_in_given_setting(total_name))
			new_elem = (total_name, epoch_name)
			comparisons_elements.append(new_elem)

		save_comparisons_models(comparisons_elements, PLOTS_HEATMAP_LATENT_SPACE_SIZE_DIRECTORY, "model_size " + current_model)

def save_comparisons_models(comparisons_elements, directory, setting_name):
	size = len(comparisons_elements)

	data = get_values_comparisons(size, comparisons_elements)

	# todo detect if only one element differs
	# row_labels = ["real images"] + [elem[0] + "\n" + elem[1] for elem in comparisons_elements]
	# col_labels = [elem[0] + "\n" + elem[1] for elem in comparisons_elements]
	row_labels = ["real images"] + [elem[0] + elem[1] for elem in comparisons_elements]
	col_labels = [elem[0] + elem[1] for elem in comparisons_elements]

	fig, ax = plt.subplots()
	im = ax.imshow(data, cmap = 'gray', interpolation = 'nearest')

	ax.set_xticks(np.arange(len(col_labels)))
	ax.set_yticks(np.arange(len(row_labels)))
	ax.set_xticklabels(col_labels)
	ax.set_yticklabels(row_labels)

	# plt.setp(ax.get_xticklabels(), rotation = 75, ha = "right", rotation_mode = "anchor")
	plt.setp(ax.get_xticklabels(), rotation = 90, ha = "right", rotation_mode = "anchor")

	plt.colorbar(im)
	plt.title("Heatmap for  " + setting_name)
	ax.set_xlabel("Discriminator")
	ax.set_ylabel("Generator")

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
	plt.savefig(Path(directory) / f"{setting_name}.png", dpi = 300)
	plt.close()

if __name__ == "__main__":
	_generate_combined_statistics_plots()
