import os
import time

import config
from config import GUI_tkinter
from ganalyzer.GUITkinter import GUITkinter
from ganalyzer.misc import get_all_models, get_available_epochs, project_array
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np

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

def get_inside_values(generator, discriminator, inpt):
	"""
	result = generator.predict(inpt)[0, :, :, :]
	generated = np.round(project_array(result, 254, -1, 1)).astype(np.uint8).tolist()

	generated_resized = np.array([result.astype(np.float64)])
	prediction_discriminator = discriminator.predict(generated_resized)[0][0]
	"""
	result = {"generator": [], "discriminator": []}

	for i in range(len(generator.layers)):
		layer_name = generator.layers[i].name
		print("==> generator", layer_name)
		result["generator"].append((i, layer_name, [[12.3, 3], [4, 5]]))

	for j in range(len(discriminator.layers)):
		layer_name = discriminator.layers[j].name
		print("==> discriminator", layer_name)
		result["discriminator"].append((j, layer_name, [[32.3, 3], [4, 5]]))

	"""
	if not self.current_model:
		logger.warning("Cannot refresh layer visualization without a model.")
		return
	
	try:
		index_layer = self.get_current_layer_index()
		layer = self.current_model.layers[index_layer]
	except (ValueError, IndexError):
		logger.warning("Invalid layer selected for visualization.")
		return
	
	try:
		intermediate = tf.keras.Model(inputs = self.current_model.inputs, outputs = layer.output)
		layer_output = intermediate.predict(self.current_input)
	except Exception:  # pragma: no cover - defensive log
		logger.exception("Failed to compute intermediate output for layer %s", layer.name)
		return
	
	if layer_output.ndim == 4:
		representation = self.get_array_representation(layer_output[0])
		is_color = representation.ndim == 3 and representation.shape[-1] in {3, 4}
	elif layer_output.ndim == 2:
		representation = self.get_rectangle_representation(layer_output[0])
		is_color = False
	else:
		logger.warning("Unsupported output shape %s for visualization", layer_output.shape)
		return
	
	self.refresh_tk_image(representation, is_color = is_color, tk_image = self.image_inside_data)
	
	"""
	# todo
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
			"discriminator_layers": ["input", "disc1", "disc2", "disc3", "out"],  # todo
			"generator_layers": ["input", "gen1", "gen2", "gen3", "out"],  # todo
		})

	@app.route("/get-result-generator", methods = ["POST"])
	def get_result_from_generator():
		data = request.get_json()
		vector = data.get("vector", [])

		inpt = np.array([vector]).astype(np.float32)

		result = generators_list[current_generator_index].predict(inpt)[0, :, :, :]
		generated = np.round(project_array(result, 254, -1, 1)).astype(np.uint8).tolist()

		generated_resized = np.array([result.astype(np.float64)])
		prediction_discriminator = discriminators_list[current_discriminator_index].predict(generated_resized)[0][0]

		return jsonify({  # todo move all in inside values
			"generated_image": generated,
			"inside_values": get_inside_values(generators_list[current_generator_index], discriminators_list[current_generator_index], inpt),
			"result_discriminator": str(prediction_discriminator)
		})

	@app.route("/change-epoch-generator", methods = ["POST"])
	def change_epoch_generator():
		print("change gen")
		data = request.get_json()
		epoch_to_look = int(data.get("new_epoch", []))
		epoch_found = get_closest_model_loaded_index(epoch_to_look, generators_list)
		global current_generator_index
		current_generator_index = epoch_found
		return jsonify({"new_epoch_found": epoch_found})

	@app.route("/change-epoch-discriminator", methods = ["POST"])
	def change_epoch_discriminator():
		print("change disc")
		data = request.get_json()
		epoch_to_look = int(data.get("new_epoch", []))
		epoch_found = get_closest_model_loaded_index(epoch_to_look, discriminators_list)
		global current_discriminator_index
		current_discriminator_index = epoch_found
		return jsonify({"new_epoch_found": epoch_found})

	app.run(debug = True)
