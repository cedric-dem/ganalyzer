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
generator_list = get_all_models("generator", available_epochs)
discriminators_list = get_all_models("discriminator", available_epochs)
t1 = time.time()
print("==> Time taken to load : ", round(t1 - t0, 2))

print("==> Number of loaded models : ", len(generator_list))

if GUI_tkinter:
	main_gui = GUITkinter(generator_list, discriminators_list)
else:
	app = Flask(__name__)
	CORS(app)

	@app.route("/get-result-generator", methods = ["POST"])
	def getResultFromGenerator():
		data = request.get_json()
		vector = data.get("vector", [])

		inpt = np.array([vector]).astype(np.float32)

		result = generator_list[-1].predict(inpt)[0, :, :, :]
		result_r = np.round(project_array(result, 254, -1, 1)).astype(np.uint8).tolist()

		#result = [[[random.randint(0,255),random.randint(0,255),random.randint(0,255)] for i in range(114)] for j in range(114)]

		return jsonify({"generated_image": result_r})

	app.run(debug = True)
