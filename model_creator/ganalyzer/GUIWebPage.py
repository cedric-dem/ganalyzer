import os
import time

import config
from ganalyzer.misc import get_all_models, get_available_epochs, project_array
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import tensorflow as tf

def _configure_model_paths(model_name, latent_space_size):
	config.model_name = model_name
	config.latent_dimension_generator = latent_space_size
	config.model_path = os.path.join(
		config.models_root_path,
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

def get_layers_list(model):
	list_layers = model.layers
	result = []
	for i in range(len(list_layers)):
		result.append(str(i) + ") " + list_layers[i].name)
	return result

class GUIWebPage(object):
	def __init__(self):
		self.generators_list = None
		self.discriminators_list = None

		self.current_generator_index = -1
		self.current_discriminator_index = -1

		app = Flask(__name__)
		CORS(app)

		@app.route("/sync-server", methods = ["POST"])
		def synchronize_server_with_client():
			print("sync server")
			data = request.get_json()

			model_size_synced = str(data.get("model_size", []))
			latent_space_size_synced = int(data.get("latent_space_size", []))
			latent_space_size_synced_str = "-ls_" + (4 - len(str(latent_space_size_synced))) * "0" + str(latent_space_size_synced)

			t0 = time.time()
			self.generators_list, self.discriminators_list = get_models_generator_and_discriminator(model_size_synced, latent_space_size_synced)
			t1 = time.time()
			models_quantity = len(self.generators_list)
			print("==> Time taken to load : ", round(t1 - t0, 2))
			print("==> Number of loaded models : ", models_quantity)

			print('====> synced with data', model_size_synced, latent_space_size_synced_str)

			return jsonify({
				"discriminator_layers": get_layers_list(self.discriminators_list[0]),
				"generator_layers": get_layers_list(self.generators_list[0]),
				"number_of_models": models_quantity,
			})

		@app.route("/get-model-prediction", methods = ["POST"])
		def get_model_prediction():

			data = request.get_json()

			vector = data.get("input_data", [])
			layer_name = data.get("layer_name", [])
			which_model = data.get("which_model", [])

			output_values = get_value_at_given_layer(self.generators_list, self.discriminators_list, self.current_generator_index, self.current_discriminator_index, vector, layer_name, which_model)

			#print('*********\n\n shape input',which_model, shape(vector), "shape output", shape(output_values))
			return jsonify({"output_values": output_values})

		@app.route("/change-epoch", methods = ["POST"])  # todo merge both change epoch in one endpoint
		def change_epoch():
			print("change epoch")
			data = request.get_json()

			epoch_to_look = int(data.get("new_epoch", []))
			which_model = data.get("which_model", [])

			if which_model == "generator":
				epoch_found = get_closest_model_loaded_index(epoch_to_look, self.generators_list)
				self.current_generator_index = epoch_found

			elif which_model == "discriminator":
				epoch_found = get_closest_model_loaded_index(epoch_to_look, self.discriminators_list)
				self.current_discriminator_index = epoch_found

			else:
				epoch_found = 0
				print("error 403", which_model)

			print('==> change epoch : ', which_model, epoch_to_look, " ( ", epoch_found, ")")
			return jsonify({"new_epoch_found": epoch_found})

		app.run(debug = True)
