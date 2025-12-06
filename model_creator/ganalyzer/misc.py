from __future__ import annotations

import os
from typing import List, Optional

import keras
import numpy as np

from config import load_quantity_gui, models_directory, results_root_path

def get_generator_model_path_at_given_epoch(epoch):
	return get_model_path_at_given_epoch("generator", epoch)

def get_discriminator_model_path_at_given_epoch(epoch):
	return get_model_path_at_given_epoch("discriminator", epoch)

def _model_directory_for(model_name: str, latent_space_size: int) -> str:
	return os.path.join(
		results_root_path,
		f"{model_name}-ls_{latent_space_size:04d}",
		"models",
	)

def get_model_path_at_given_epoch(model_type, epoch, models_dir: Optional[str] = None):
	filename = f"{model_type}_epoch_{epoch:06d}.keras"
	current_models_directory = models_dir or models_directory
	return os.path.join(current_models_directory, filename)

def get_model_path_at_given_epoch_closest_possible(model_type, epoch, available_epochs, models_dir: Optional[str] = None):
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

def get_available_epochs(models_dir: Optional[str] = None):
	models_list = get_list_of_keras_models(models_dir)
	return [int(model.split("_")[-1].split(".")[0]) for model in models_list if model.startswith("discriminator")]

def _indexes_to_load(models_quantity):
	if models_quantity == 0:
		return []

	if load_quantity_gui >= models_quantity:
		return list(range(models_quantity))

	take_every = max(1, models_quantity // load_quantity_gui)
	indexes = set(range(0, models_quantity, take_every))
	indexes.update({0, models_quantity - 1})
	return sorted(indexes)

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

def get_list_of_keras_models(models_dir: Optional[str] = None):
	current_models_directory = models_dir or models_directory

	if not os.path.isdir(current_models_directory):
		return []

	complete_list = sorted(os.listdir(current_models_directory))
	return [filename for filename in complete_list if not filename.endswith(".csv")]

def get_current_epoch(models_dir: Optional[str] = None):
	keras_models = get_list_of_keras_models(models_dir)
	if not keras_models:
		return 0
	return int(keras_models[-1].split("_")[-1].split(".")[0])
