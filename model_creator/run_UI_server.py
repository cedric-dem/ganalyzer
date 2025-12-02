import time

from config import GUI_tkinter
from ganalyzer.GUITkinter import GUITkinter
from ganalyzer.misc import get_all_models, get_available_epochs, project_array
from flask import Flask, jsonify, request
from flask_cors import CORS
import random
import numpy as np

# load all the models
t0 = time.time()
available_epochs = get_available_epochs()
generators_list = get_all_models("generator", available_epochs)
discriminators_list = get_all_models("discriminator", available_epochs)
t1 = time.time()
print("==> Time taken to load : ", round(t1 - t0, 2))

print("==> Number of loaded models : ", len(generators_list))

models_quantity = len(generators_list)

global current_generator, current_discriminator

current_generator = None
current_discriminator = None

def get_closest_model_loaded_index(model_index, models_list):
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

if GUI_tkinter:
	main_gui = GUITkinter(generators_list, discriminators_list)
else:
	app = Flask(__name__)
	CORS(app)

	current_generator = generators_list[-1]
	current_discriminator = discriminators_list[-1]

	@app.route("/get-result-generator", methods = ["POST"])
	def getResultFromGenerator():
		data = request.get_json()
		vector = data.get("vector", [])

		inpt = np.array([vector]).astype(np.float32)

		result = current_generator.predict(inpt)[0, :, :, :]
		generated = np.round(project_array(result, 254, -1, 1)).astype(np.uint8).tolist()

		generated_resized = np.array([result.astype(np.float64)])
		prediction_discriminator = current_discriminator.predict(generated_resized)[0][0]

		return jsonify({
			"generated_image": generated,
			"result_discriminator": str(prediction_discriminator)
		})

	@app.route("/change-epoch-generator", methods = ["POST"])
	def changeEpochGenerator():
		print("change gen")
		data = request.get_json()
		epoch_to_look = int(data.get("new_epoch", []))
		epoch_found = get_closest_model_loaded_index(epoch_to_look, generators_list)
		global current_generator
		current_generator = generators_list[epoch_found]
		return jsonify({"new_epoch_found": epoch_found})

	@app.route("/change-epoch-discriminator", methods = ["POST"])
	def changeEpochDiscriminator():
		print("change disc")
		data = request.get_json()
		epoch_to_look = int(data.get("new_epoch", []))
		epoch_found = get_closest_model_loaded_index(epoch_to_look, discriminators_list)
		global current_discriminator
		current_discriminator = discriminators_list[epoch_found]
		return jsonify({"new_epoch_found": epoch_found})

	app.run(debug = True)
