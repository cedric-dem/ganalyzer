import os
import time

import config
from config import GUI_tkinter
from ganalyzer.GUITkinter import GUITkinter
from ganalyzer.misc import get_all_models, get_available_epochs, project_array
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import tensorflow as tf

def _configure_model_paths(model_name, latent_space_size):
	config.model_name = model_name
	config.latent_dimension_generator = latent_space_size
	config.model_path = os.path.join(
		config.results_root_path,
		f"{model_name}-ls_{latent_space_size:04d}",
	)
	config.models_directory = os.path.join(config.model_path, "models")
	os.makedirs(config.models_directory, exist_ok = True)

def get_models_generator_and_discriminator(model_name, latent_space_size):  # TOdo : remove code duplication
	_configure_model_paths(model_name, latent_space_size)

	available_epochs = get_available_epochs(config.models_directory)
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

# load all the models

global generators_list, discriminators_list
generators_list, discriminators_list = None, None

global current_generator_index, current_discriminator_index
current_generator_index, current_discriminator_index = None, None

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

def get_value_at_given_layer(vector, layer_name, which_model):
	layer_index = int(layer_name.split(")")[0])
	# print("===*****************", vector, layer_name, which_model, " l_index:", layer_index)

	if which_model == "generator":
		model = generators_list[current_generator_index]
		intermediate = tf.keras.Model(inputs = model.inputs, outputs = model.layers[layer_index].output)

		inpt = np.array([vector]).astype(np.float32)  # todo isolate in a function

		layer_output_raw = intermediate.predict(inpt)

		layer_output = np.round(project_array(layer_output_raw, 254, -1, 1)).tolist()[0]  # todo isolate in a function
		# layer_output = np.round(project_array(layer_output_raw, 254, -1, 1)).astype(np.float32).tolist()  # todo isolate in a function
		# could do more work here, to prepare for front end
		pass

	elif which_model == "discriminator":
		model = discriminators_list[current_discriminator_index]
		intermediate = tf.keras.Model(inputs = model.inputs, outputs = model.layers[layer_index].output)

		inpt = np.array([np.array(vector).astype(np.float64)])  # todo isolate in function

		layer_output = intermediate.predict(inpt).tolist()[0]

	else:
		raise ValueError("Unknown model type.")

	return layer_output

def get_layers_list(model):
	list_layers = model.layers
	result = []
	for i in range(len(list_layers)):
		result.append(str(i) + ") " + list_layers[i].name)
	return result

if GUI_tkinter:
	main_gui = GUITkinter(generators_list, discriminators_list)
else:
	app = Flask(__name__)
	CORS(app)

	current_generator_index = -1
	current_discriminator_index = -1

	@app.route("/sync-server", methods = ["POST"])
	def synchronize_server_with_client():
		print("sync server")
		data = request.get_json()

		model_size_synced = str(data.get("model_size", []))
		latent_space_size_synced = int(data.get("latent_space_size", []))
		latent_space_size_synced_str = "-ls_" + (4 - len(str(latent_space_size_synced))) * "0" + str(latent_space_size_synced)

		global generators_list, discriminators_list

		t0 = time.time()
		generators_list, discriminators_list = get_models_generator_and_discriminator(model_size_synced, latent_space_size_synced)
		t1 = time.time()
		models_quantity = len(generators_list)
		print("==> Time taken to load : ", round(t1 - t0, 2))
		print("==> Number of loaded models : ", models_quantity)

		print('====> synced with data', model_size_synced, latent_space_size_synced_str)

		return jsonify({
			"discriminator_layers": get_layers_list(discriminators_list[0]),
			"generator_layers": get_layers_list(generators_list[0]),
		})

	def get_output_generator(vector):
		inpt = np.array([vector]).astype(np.float32)
		result = generators_list[current_generator_index].predict(inpt)[0, :, :, :]
		return result

	@app.route("/get-result-generator", methods = ["POST"])  # todo delete this function, can be replace d by get_inside_values with last layer
	def get_result_from_generator():
		data = request.get_json()
		vector = data.get("vector", [])

		result = get_output_generator(vector)

		generated = np.round(project_array(result, 254, -1, 1)).astype(np.uint8).tolist()
		generated_resized = np.array([result.astype(np.float64)])

		prediction_discriminator = discriminators_list[current_discriminator_index].predict(generated_resized)[0][0]

		return jsonify({  # todo move all in inside values
			"generated_image": generated,
			"result_discriminator": str(prediction_discriminator)
		})

	@app.route("/get-inside-values", methods = ["POST"])
	def get_inside_values():
		data = request.get_json()
		vector = data.get("vector", [])
		layer_name = data.get("layer_name", [])
		which_model = data.get("which_model", [])

		# blabla
		# print('====> ins values \n\n', vector, layer_name, which_model)
		inside_values = get_value_at_given_layer(vector, layer_name, which_model)

		return jsonify({
			"inside_values": inside_values
		})

	@app.route("/change-epoch-generator", methods = ["POST"])
	def change_epoch_generator():
		print("change gen")
		data = request.get_json()
		epoch_to_look = int(data.get("new_epoch", []))
		epoch_found = get_closest_model_loaded_index(epoch_to_look, generators_list)
		global current_generator_index
		current_generator_index = epoch_found
		print('====In epoch :', epoch_to_look, " .. ", epoch_found)
		return jsonify({"new_epoch_found": epoch_found})

	@app.route("/change-epoch-discriminator", methods = ["POST"])
	def change_epoch_discriminator():
		print("change disc")
		data = request.get_json()
		epoch_to_look = int(data.get("new_epoch", []))
		epoch_found = get_closest_model_loaded_index(epoch_to_look, discriminators_list)
		global current_discriminator_index
		current_discriminator_index = epoch_found
		print('====In epoch :', epoch_to_look, " .. ", epoch_found)
		return jsonify({"new_epoch_found": epoch_found})

	app.run(debug = True)
