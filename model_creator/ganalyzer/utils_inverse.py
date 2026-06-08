from __future__ import annotations

from config import (
	AUTOENCODER_PIXEL_DIFFERENCE_BAR_COUNT,
	EPOCH_GLOBAL_NAME,
	GENERATOR_GLOBAL_NAME,
	IMAGE_NORMALIZATION_CENTER,
	IMAGE_NORMALIZATION_SCALE,
	INVERSE_GENERATOR_DIRECTORY_NAME,
	INVERSE_GENERATOR_MODEL_TYPE,
	IS_RGB_IMAGES,
	MODELS_DIRECTORY,
	MODELS_DIRECTORY_NAME,
	MODELS_DIRECTORY_NAME_INVERSE,
	NUMBER_COMPARISON,
	STATISTICS_CSV_FILENAME,
	STR_PATH_DATASET,
	INVERSE_PLOTS_DIRECTORY_NAME,
	INVERSE_COMPARISON_DIRECTORY_NAME,
	INVERSE_GENERATOR_PLOT_IMAGE_NAMES,
	INVERSE_GENERATOR_PLOT_NAMES,
	INVERSE_GENERATOR_X_LABEL_NAMES,
	INVERSE_GENERATOR_Y_LABEL_NAMES,
	LATENT_DIMENSION_GENERATOR,
)

from tensorflow import keras
import csv
import shutil
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_model_filename(model_type, epoch):
	return f"{model_type}_{EPOCH_GLOBAL_NAME}_{epoch:06d}.keras"

def get_model_path_at_given_epoch(model_type, epoch, models_dir):
	return Path(models_dir) / get_model_filename(model_type, epoch)

def get_inverse_generator_plots_directory(generator_epoch):
	return get_inverse_generator_run_directory(generator_epoch) / INVERSE_PLOTS_DIRECTORY_NAME

def get_inverse_generator_model_path(generator_epoch, inverse_generator_epoch):
	return get_inverse_generator_models_directory(generator_epoch) / get_model_filename(INVERSE_GENERATOR_MODEL_TYPE, inverse_generator_epoch)

def get_inverse_generator_models_directory(generator_epoch):
	return get_inverse_generator_run_directory(generator_epoch) / MODELS_DIRECTORY_NAME_INVERSE

def get_inverse_generator_run_directory(generator_epoch):
	return Path(MODELS_DIRECTORY).parent / INVERSE_GENERATOR_DIRECTORY_NAME / f"{MODELS_DIRECTORY_NAME}_{INVERSE_GENERATOR_MODEL_TYPE}_{generator_epoch}"

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

def produce_autoencoder_assets():
	generator_epoch = get_last_model_epoch(GENERATOR_GLOBAL_NAME, MODELS_DIRECTORY)
	generator_path = get_model_path_at_given_epoch(GENERATOR_GLOBAL_NAME, generator_epoch, MODELS_DIRECTORY)
	inverse_generator_models_directory = get_inverse_generator_models_directory(generator_epoch)
	inverse_generator_epoch = get_last_model_epoch(INVERSE_GENERATOR_MODEL_TYPE, inverse_generator_models_directory)
	inverse_generator_path = get_inverse_generator_model_path(generator_epoch, inverse_generator_epoch)
	comparison_directory = get_inverse_generator_run_directory(generator_epoch) / INVERSE_COMPARISON_DIRECTORY_NAME

	plots_directory = get_inverse_generator_plots_directory(generator_epoch)

	print("==> loading generator from : ", generator_path)
	generator = keras.models.load_model(str(generator_path))
	generator.trainable = False

	print("==> loading inverse generator from : ", inverse_generator_path)
	inverse_generator = keras.models.load_model(str(inverse_generator_path))
	inverse_generator.trainable = False

	image_paths = get_autoencoder_comparison_image_paths()
	if not image_paths:
		raise ValueError(f"No dataset images available in {STR_PATH_DATASET}.")

	recreate_directory(comparison_directory)
	print(f"==> writing {len(image_paths)} autoencoder comparison images to : {comparison_directory}")
	pixel_differences = []

	for image_index, image_path in enumerate(image_paths, start = 1):
		print(f"=> producing autoencoder comparison {image_index}/{len(image_paths)} from {image_path}")
		normalized_image = load_image(image_path)
		model_input = image_to_model_batch(normalized_image)
		latent_vector = inverse_generator(model_input, training = False)
		reconstructed_image = generator(latent_vector, training = False).numpy()[0]

		pixel_differences.extend(compute_pixel_differences(normalized_image, reconstructed_image))
		comparison_image = make_side_by_side_image(normalized_image, reconstructed_image)
		comparison_image.save(comparison_directory / f"comparison_{image_index:04d}.png", format = "PNG")

	plots_directory.mkdir(parents = True, exist_ok = True)
	barplot_path = plots_directory / INVERSE_GENERATOR_PLOT_IMAGE_NAMES["autoencoder_pixel_difference"]
	autoencoder_mae = float(np.mean(pixel_differences))
	print(f"==> autoencoder pixel mae : {autoencoder_mae:.6f}")

	print(f"==> writing autoencoder pixel difference bar plot to : {barplot_path}")
	produce_pixel_difference_barplot(pixel_differences, barplot_path)

def compute_pixel_differences(original_image, reconstructed_image):
	original_output = normalized_image_to_pil(original_image)
	reconstructed_output = normalized_image_to_pil(reconstructed_image)

	if reconstructed_output.size != original_output.size:
		reconstructed_output = reconstructed_output.resize(original_output.size, Image.Resampling.LANCZOS)
	if reconstructed_output.mode != original_output.mode:
		reconstructed_output = reconstructed_output.convert(original_output.mode)

	original_pixels = np.asarray(original_output, dtype = np.float32)
	reconstructed_pixels = np.asarray(reconstructed_output, dtype = np.float32)
	return np.abs(original_pixels - reconstructed_pixels).reshape(-1).tolist()

def produce_pixel_difference_barplot(pixel_differences, output_path):
	if not pixel_differences:
		raise ValueError("Cannot produce a pixel difference bar plot without pixel differences.")

	bin_count = AUTOENCODER_PIXEL_DIFFERENCE_BAR_COUNT
	if bin_count <= 0:
		raise ValueError("AUTOENCODER_PIXEL_DIFFERENCE_BAR_COUNT must be greater than zero.")

	bin_edges, pixel_percentages_by_difference = compute_pixel_difference_percentages_by_bin(pixel_differences, bin_count)

	fig, ax = plt.subplots(figsize = (14, 5))
	ax.bar(bin_edges[:-1], pixel_percentages_by_difference, width = np.diff(bin_edges), align = "edge")
	ax.set_title(INVERSE_GENERATOR_PLOT_NAMES["autoencoder_pixel_difference"])
	ax.set_xlabel(INVERSE_GENERATOR_X_LABEL_NAMES["autoencoder_pixel_difference"])
	ax.set_ylabel(INVERSE_GENERATOR_Y_LABEL_NAMES["autoencoder_pixel_difference"])
	ax.set_xlim(0, 100)
	ax.set_xticks(np.arange(0, 101, 10))
	ax.grid(axis = "y", linestyle = "--", alpha = 0.4)
	fig.tight_layout()
	fig.savefig(output_path, dpi = 300)
	plt.close(fig)

def compute_pixel_difference_percentages_by_bin(pixel_differences, bin_count):
	pixel_difference_percentages = np.asarray(pixel_differences, dtype = np.float32) / 255.0 * 100.0
	bin_edges = np.linspace(0, 100, bin_count + 1)
	bin_counts, _ = np.histogram(pixel_difference_percentages, bins = bin_edges)
	pixel_percentages_by_difference = bin_counts / len(pixel_difference_percentages) * 100.0
	return bin_edges, pixel_percentages_by_difference

def produce_pixel_difference_boxplot(pixel_differences, output_path):
	produce_pixel_difference_barplot(pixel_differences, output_path)

def get_autoencoder_comparison_image_paths():
	dataset_directory = Path(STR_PATH_DATASET)
	if not dataset_directory.is_dir():
		raise ValueError(f"Dataset directory does not exist: {dataset_directory}")

	image_paths = [path for path in sorted(dataset_directory.iterdir()) if path.is_file()]
	return image_paths[:NUMBER_COMPARISON]

def recreate_directory(directory):
	if directory.exists():
		shutil.rmtree(directory)
	directory.mkdir(parents = True, exist_ok = True)

def load_image(image_path):
	with Image.open(image_path) as image:
		if IS_RGB_IMAGES:
			image = image.convert("RGB")
		else:
			image = image.convert("L")
		image_array = np.asarray(image, dtype = np.float32)

	return (image_array - IMAGE_NORMALIZATION_CENTER) / IMAGE_NORMALIZATION_SCALE

def image_to_model_batch(image):
	image_array = np.asarray(image, dtype = np.float32)
	if image_array.ndim == 2:
		image_array = np.expand_dims(image_array, axis = -1)
	return np.expand_dims(image_array, axis = 0)

def make_side_by_side_image(original_image, reconstructed_image):
	original_output = normalized_image_to_pil(original_image)
	reconstructed_output = normalized_image_to_pil(reconstructed_image)

	if reconstructed_output.size != original_output.size:
		reconstructed_output = reconstructed_output.resize(original_output.size, Image.Resampling.LANCZOS)
	if reconstructed_output.mode != original_output.mode:
		reconstructed_output = reconstructed_output.convert(original_output.mode)

	comparison = Image.new(original_output.mode, (original_output.width + reconstructed_output.width, original_output.height))
	comparison.paste(original_output, (0, 0))
	comparison.paste(reconstructed_output, (original_output.width, 0))
	return comparison

def normalized_image_to_pil(image):
	image_array = np.asarray(image)

	if not IS_RGB_IMAGES and image_array.ndim == 3 and image_array.shape[-1] == 3:
		image_array = np.mean(image_array, axis = -1, keepdims = True)

	image_array = np.clip((image_array + 1.0) * IMAGE_NORMALIZATION_CENTER, 0, 255).astype(np.uint8)

	if image_array.ndim == 3 and image_array.shape[-1] == 1:
		image_array = np.squeeze(image_array, axis = -1)

	return Image.fromarray(image_array)

def produce_generator_inverse_and_generator_assets():
	"""Produce latent-vector round-trip assets for generator -> inverse generator.

	Random latent vectors are passed through the generator, then through the
	inverse generator. The absolute differences between the original and
	recovered latent vectors are aggregated and written as a distribution plot.
	"""
	generator_epoch = get_last_model_epoch(GENERATOR_GLOBAL_NAME, MODELS_DIRECTORY)
	generator_path = get_model_path_at_given_epoch(GENERATOR_GLOBAL_NAME, generator_epoch, MODELS_DIRECTORY)
	inverse_generator_models_directory = get_inverse_generator_models_directory(generator_epoch)
	inverse_generator_epoch = get_last_model_epoch(INVERSE_GENERATOR_MODEL_TYPE, inverse_generator_models_directory)
	inverse_generator_path = get_inverse_generator_model_path(generator_epoch, inverse_generator_epoch)
	plots_directory = get_inverse_generator_plots_directory(generator_epoch)

	print("==> loading generator from : ", generator_path)
	generator = keras.models.load_model(str(generator_path))
	generator.trainable = False

	print("==> loading inverse generator from : ", inverse_generator_path)
	inverse_generator = keras.models.load_model(str(inverse_generator_path))
	inverse_generator.trainable = False

	latent_differences = []
	for comparison_index in range(1, NUMBER_COMPARISON + 1):
		print(f"=> producing generator/inverse comparison {comparison_index}/{NUMBER_COMPARISON}")
		latent_vector = np.random.normal(0.0, 1.0, size = (1, LATENT_DIMENSION_GENERATOR)).astype(np.float32)
		generated_image = generator(latent_vector, training = False)
		reconstructed_latent_vector = inverse_generator(generated_image, training = False)
		latent_differences.extend(compute_latent_vector_differences(latent_vector, reconstructed_latent_vector))

	plots_directory.mkdir(parents = True, exist_ok = True)
	barplot_path = plots_directory / INVERSE_GENERATOR_PLOT_IMAGE_NAMES["generator_inverse_latent_difference"]
	latent_mae = float(np.mean(latent_differences))
	print(f"==> generator/inverse latent mae : {latent_mae:.6f}")

	print(f"==> writing generator/inverse latent difference bar plot to : {barplot_path}")
	produce_latent_difference_barplot(latent_differences, barplot_path)

def compute_latent_vector_differences(latent_vector, reconstructed_latent_vector):
	latent_values = np.asarray(latent_vector, dtype = np.float32)
	reconstructed_latent_values = np.asarray(reconstructed_latent_vector, dtype = np.float32)
	return np.abs(latent_values - reconstructed_latent_values).reshape(-1).tolist()

def produce_latent_difference_barplot(latent_differences, output_path):
	if not latent_differences:
		raise ValueError("Cannot produce a latent difference bar plot without latent differences.")

	bin_count = AUTOENCODER_PIXEL_DIFFERENCE_BAR_COUNT
	if bin_count <= 0:
		raise ValueError("AUTOENCODER_PIXEL_DIFFERENCE_BAR_COUNT must be greater than zero.")

	latent_differences = np.asarray(latent_differences, dtype = np.float32)
	upper_bound = float(np.max(latent_differences))
	if upper_bound == 0.0:
		upper_bound = 1.0

	bin_edges = np.linspace(0, upper_bound, bin_count + 1)
	bin_counts, _ = np.histogram(latent_differences, bins = bin_edges)
	latent_percentages_by_difference = bin_counts / len(latent_differences) * 100.0

	fig, ax = plt.subplots(figsize = (14, 5))
	ax.bar(bin_edges[:-1], latent_percentages_by_difference, width = np.diff(bin_edges), align = "edge")
	ax.set_title(INVERSE_GENERATOR_PLOT_NAMES["generator_inverse_latent_difference"])
	ax.set_xlabel(INVERSE_GENERATOR_X_LABEL_NAMES["generator_inverse_latent_difference"])
	ax.set_ylabel(INVERSE_GENERATOR_Y_LABEL_NAMES["generator_inverse_latent_difference"])
	ax.set_xlim(0, upper_bound)
	ax.grid(axis = "y", linestyle = "--", alpha = 0.4)
	fig.tight_layout()
	fig.savefig(output_path, dpi = 300)
	plt.close(fig)

def produce_plots_inverse():
	generator_epoch = get_last_model_epoch(GENERATOR_GLOBAL_NAME, MODELS_DIRECTORY)
	inverse_generator_run_directory = get_inverse_generator_run_directory(generator_epoch)
	statistics_path = inverse_generator_run_directory / STATISTICS_CSV_FILENAME
	plots_directory = inverse_generator_run_directory / INVERSE_PLOTS_DIRECTORY_NAME

	epochs, loss_mses, loss_maes = load_inverse_generator_plot_statistics(statistics_path)

	plots_directory.mkdir(parents = True, exist_ok = True)
	loss_mse_output_path = plots_directory / INVERSE_GENERATOR_PLOT_IMAGE_NAMES["inverse_loss_mse"]
	print("==> writing inverse generator loss_mse plot to : ", loss_mse_output_path)
	plot_inverse_generator_loss_mse(epochs, loss_mses, loss_mse_output_path)

	loss_mae_output_path = plots_directory / INVERSE_GENERATOR_PLOT_IMAGE_NAMES["inverse_loss_mae"]
	print("==> writing inverse generator loss_mae plot to : ", loss_mae_output_path)
	plot_inverse_generator_loss_mae(epochs, loss_maes, loss_mae_output_path)

def load_inverse_generator_loss_statistics(statistics_path):
	epochs, loss_mses, _ = load_inverse_generator_plot_statistics(statistics_path, require_loss_mae = False)
	return epochs, loss_mses

def load_inverse_generator_plot_statistics(statistics_path, require_loss_mae = True):
	statistics_path = Path(statistics_path)
	if not statistics_path.is_file():
		raise ValueError(f"Inverse generator statistics file does not exist: {statistics_path}")

	epochs = []
	loss_mses = []
	loss_maes = []

	with statistics_path.open(newline = "", encoding = "utf-8") as statistics_file:
		reader = csv.DictReader(statistics_file)
		if not reader.fieldnames:
			raise ValueError(f"Inverse generator statistics file is empty: {statistics_path}")

		epoch_column = find_csv_column(reader.fieldnames, ("epoch_id", "epoch"))
		loss_mse_column = find_csv_column(reader.fieldnames, ("loss_mse", "latent_mse", "loss", "mse"))
		loss_mae_column = find_csv_column(reader.fieldnames, ("loss_mae", "latent_mae", "mae", "autoencoder_mae"))
		if epoch_column is None:
			raise ValueError(f"Inverse generator statistics file has no epoch column: {statistics_path}")
		if loss_mse_column is None:
			raise ValueError(f"Inverse generator statistics file has no loss_mse column: {statistics_path}")
		if require_loss_mae and loss_mae_column is None:
			raise ValueError(f"Inverse generator statistics file has no loss_mae column: {statistics_path}")

		for row in reader:
			epoch_value = row[epoch_column].strip()
			loss_mse_value = row[loss_mse_column].strip()
			loss_mae_value = row[loss_mae_column].strip() if loss_mae_column is not None else ""
			if not epoch_value or not loss_mse_value or (require_loss_mae and not loss_mae_value):
				continue

			epochs.append(int(float(epoch_value)))
			loss_mses.append(float(loss_mse_value))
			if loss_mae_value:
				loss_maes.append(float(loss_mae_value))

	if not epochs:
		raise ValueError(f"Inverse generator statistics file has no data rows: {statistics_path}")

	return epochs, loss_mses, loss_maes

def find_csv_column(fieldnames, candidate_names):
	normalized_candidates = {candidate_name.lower() for candidate_name in candidate_names}

	for fieldname in fieldnames:
		if fieldname.lower() in normalized_candidates:
			return fieldname

	return None

def plot_inverse_generator_loss_mse(epochs, loss_mses, output_path):
	plt.figure(figsize = (12, 6))
	plt.plot(epochs, loss_mses, label = "loss_mse")
	plt.title(INVERSE_GENERATOR_PLOT_NAMES["inverse_loss_mse"])
	plt.xlabel(INVERSE_GENERATOR_X_LABEL_NAMES["inverse_loss_mse"])
	plt.ylabel(INVERSE_GENERATOR_Y_LABEL_NAMES["inverse_loss_mse"])
	plt.ylim(0, 1)  # Set y-axis range from 0 to 1
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_path, format = "png")
	plt.close()

def plot_inverse_generator_loss_mae(epochs, loss_maes, output_path):
	plt.figure(figsize = (12, 6))
	plt.plot(epochs, loss_maes, label = "loss_mae")
	plt.title(INVERSE_GENERATOR_PLOT_NAMES["inverse_loss_mae"])
	plt.xlabel(INVERSE_GENERATOR_X_LABEL_NAMES["inverse_loss_mae"])
	plt.ylabel(INVERSE_GENERATOR_Y_LABEL_NAMES["inverse_loss_mae"])
	plt.ylim(0, 0.2)
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_path, format = "png")
	plt.close()
