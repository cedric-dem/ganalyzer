import csv
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (PLOTS_ROOT_DIRECTORY, every_models_statistics_path)
from ganalyzer.model_config import all_models

STATISTICS_FILENAME = "statistics.csv"
PLOTS_ROOT_DIRECTORY_STR = str(PLOTS_ROOT_DIRECTORY)

def _parse_float(value):
	if value is None:
		return None
	value = value.strip()
	if not value:
		return None
	try:
		return float(value)
	except ValueError:
		return None

@dataclass
class Statistics:
	training_loss: List[float]
	validation_loss: List[float]
	generator_loss: List[float]
	discriminator_loss: List[float]
	epoch_durations: List[float]

def _load_statistics(csv_path):
	training_loss = []
	validation_loss = []
	generator_loss = []
	discriminator_loss = []
	epoch_durations = []

	with open(csv_path, newline = "", encoding = "utf-8") as csv_file:
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

def _plot_loss_series(series, output_path, title, xlabel = "Epoch", ylabel = "Loss"):
	plt.figure()

	plotted_any = False
	show_legend = False

	for label, values in series:
		if not values:
			continue

		epochs = range(1, len(values) + 1)

		plt.plot(epochs, values, label = label, color = get_color(label))
		show_legend = True
		plotted_any = True

	if not plotted_any:
		plt.close()
		return False

	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	if show_legend:
		plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(output_path, format = "jpg")
	plt.close()
	return True

def get_color(label):
	return proportion_to_color(all_models.index(label) / len(all_models))

def proportion_to_color(p):
	p = max(0.0, min(1.0, p))
	return f"#{int(255 * (1 - p)):02x}{int(255 * p):02x}00"

def _plot_combined_losses(stats_by_model, output_dir):
	plotted_any = False

	generator_series = []
	discriminator_series = []

	for model_name, stats in stats_by_model.items():
		if stats.generator_loss:
			generator_series.append((model_name, stats.generator_loss))
		if stats.discriminator_loss:
			discriminator_series.append((model_name, stats.discriminator_loss))

	if generator_series and _plot_loss_series(generator_series, os.path.join(output_dir, "generator_loss.jpg"), "Generator Loss Over Epochs"):
		plotted_any = True

	if discriminator_series and _plot_loss_series(discriminator_series, os.path.join(output_dir, "discriminator_loss.jpg"), "Discriminator Loss Over Epochs"):
		plotted_any = True

	return plotted_any

def _plot_combined_epoch_times(stats_by_model, output_dir):
	plt.figure()

	plotted_any = False
	for model_name, stats in stats_by_model.items():
		if not stats.epoch_durations:
			continue

		plt.plot(
			range(1, len(stats.epoch_durations) + 1),
			stats.epoch_durations,
			label = model_name,
			color = get_color(model_name)
		)
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
	plt.savefig(os.path.join(output_dir, "epoch_times.jpg"), format = "jpg")
	plt.close()
	return True

def _generate_all_models_statistics_plots():
	any_plots_generated = False

	for model_directory in every_models_statistics_path:
		csv_path = os.path.join(model_directory, STATISTICS_FILENAME)
		if not os.path.exists(csv_path):
			continue

		output_dir = os.path.join(PLOTS_ROOT_DIRECTORY_STR, os.path.basename(model_directory))
		os.makedirs(output_dir, exist_ok = True)

		if _generate_statistics_plots(csv_path, output_dir):  # TODO remove this
			any_plots_generated = True

	if not any_plots_generated:
		print("No statistics were found to generate plots.")

	return any_plots_generated

def _collect_statistics_by_model() -> dict[str, Statistics]:
	stats_by_model: dict[str, Statistics] = {}

	for model_directory in every_models_statistics_path:
		csv_path = os.path.join(model_directory, STATISTICS_FILENAME)
		if not os.path.exists(csv_path):
			continue

		stats = _load_statistics(csv_path)
		if not (stats.generator_loss or stats.discriminator_loss or stats.epoch_durations):
			continue

		model_name = os.path.basename(model_directory)
		stats_by_model[model_name] = stats

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

	os.makedirs(PLOTS_ROOT_DIRECTORY_STR, exist_ok = True)

	plotted_any = False
	if has_generator_loss or has_discriminator_loss:
		if _plot_combined_losses(stats_by_model, PLOTS_ROOT_DIRECTORY_STR):
			plotted_any = True

	if has_epoch_durations and _plot_combined_epoch_times(stats_by_model, PLOTS_ROOT_DIRECTORY_STR):
		plotted_any = True

	if plotted_any:
		print(f"Saved combined statistics plots to '{PLOTS_ROOT_DIRECTORY_STR}'.")
	else:
		print("No numeric statistics were found to plot for combined statistics.")

if __name__ == "__main__":
	_generate_combined_statistics_plots()
