from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PLOTS_ROOT_DIRECTORY, every_models_statistics_path, results_root_path
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

if __name__ == "__main__":
	_generate_combined_statistics_plots()
