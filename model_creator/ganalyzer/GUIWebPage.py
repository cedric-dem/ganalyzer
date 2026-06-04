import time

from flask import Flask, jsonify, request
from flask_cors import CORS

from ganalyzer.utils_web_ui import *

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
			models_quantity = max(len(self.generators_list), len(self.discriminators_list))
			print("==> Time taken to load : ", round(t1 - t0, 2))
			print("==> Number of loaded models : ", models_quantity)

			print('====> synced with data', model_size_synced, latent_space_size_synced_str)

			try:
				generator = get_first_loaded_model(self.generators_list, "generator")
				discriminator = get_first_loaded_model(self.discriminators_list, "discriminator")
			except ValueError as error:
				return jsonify({
					"error": str(error),
					"models_directory": config.MODELS_DIRECTORY,
					"number_of_models": models_quantity,
				}), 404

			return jsonify({
				"discriminator_layers": get_layers_list(discriminator),
				"generator_layers": get_layers_list(generator),
				"number_of_models": models_quantity,
			})

		@app.route("/get-model-prediction", methods = ["POST"])
		def get_model_prediction():

			data = request.get_json()

			vector = data.get("input_data", [])
			layer_name = data.get("layer_name", [])
			which_model = data.get("which_model", [])

			output_values = get_value_at_given_layer(self.generators_list, self.discriminators_list, self.current_generator_index, self.current_discriminator_index, vector, layer_name, which_model)

			# print('*********\n\n shape input',which_model, shape(vector), "shape output", shape(output_values))
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
