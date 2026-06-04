import os

import config
from ganalyzer.misc import get_all_models, get_available_epochs, project_array
import numpy as np
import tensorflow as tf

def _configure_model_paths(model_name, latent_space_size):
	config.MODEL_NAME = model_name
	config.LATENT_DIMENSION_GENERATOR = latent_space_size
	config.MODELS_PATH = os.path.join(
		config.STR_PATH_MODELS_ROOT,
		f"{model_name}-ls_{latent_space_size:04d}",
	)
	config.MODELS_DIRECTORY = os.path.join(config.MODELS_PATH, config.MODELS_DIRECTORY_NAME)

def get_models_generator_and_discriminator(model_name, latent_space_size):  # TOdo : remove code duplication
	_configure_model_paths(model_name, latent_space_size)

	available_epochs = get_available_epochs(config.MODELS_DIRECTORY)
	generators_list = get_all_models(
		model_type = "generator",
		available_epochs = available_epochs,
		model_name = model_name,
		latent_space_size = latent_space_size,
	)
	discriminators_list = get_all_models(
		model_type = "discriminator",
		available_epochs = available_epochs,
		model_name = model_name,
		latent_space_size = latent_space_size,
	)
	return generators_list, discriminators_list

def get_closest_model_loaded_index(model_index, models_list):
	models_quantity = len(models_list)
	if 0 <= model_index < models_quantity and models_list[model_index]:
		return model_index

	lower = model_index - 1
	upper = model_index + 1
	while lower >= 0 or upper < models_quantity:
		if lower >= 0 and models_list[lower]:
			return lower
		if upper < models_quantity and models_list[upper]:
			return upper
		lower -= 1
		upper += 1

	raise ValueError("No models available in the provided list.")

def get_value_at_given_layer(generators_list, discriminators_list, current_generator_index, current_discriminator_index, vector, layer_name, which_model):
	layer_index = int(layer_name.split(")")[0])
	# print("===*****************", vector, layer_name, which_model, " l_index:", layer_index)

	if which_model == "generator":
		model = generators_list[current_generator_index]
		intermediate = tf.keras.Model(inputs = model.inputs, outputs = model.layers[layer_index].output)

		inpt = np.array([vector[0][0]]).astype(np.float32)  # todo isolate in a function

		layer_output = np.round(project_array(intermediate.predict(inpt), 254, -1, 1)).tolist()[0]

	elif which_model == "discriminator":
		model = discriminators_list[current_discriminator_index]
		intermediate = tf.keras.Model(inputs = model.inputs, outputs = model.layers[layer_index].output)

		inpt = np.array([np.array(vector).astype(np.float64)])  # todo isolate in function

		layer_output = intermediate.predict(inpt).tolist()[0]

	else:
		raise ValueError("Unknown model type.")

	ndim = len(shape(layer_output))
	if ndim == 1:
		layer_output = [[layer_output]]
	elif ndim == 2:
		layer_output = [layer_output]
	elif ndim == 3:
		pass
	else:
		raise ValueError("number dim unknown")
	return layer_output

def shape(mat):
	if not isinstance(mat, list):
		return ()
	return (len(mat),) + shape(mat[0]) if mat else (0,)

def get_first_loaded_model(models_list, model_type):
	for model in models_list:
		if model is not None:
			return model
	raise ValueError(f"No loaded {model_type} model available.")

def get_layers_list(model):
	list_layers = model.layers
	result = []
	for i in range(len(list_layers)):
		result.append(str(i) + ") " + list_layers[i].name)
	return result
