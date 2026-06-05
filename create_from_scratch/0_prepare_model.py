import zipfile
import os
import json
from pathlib import Path
import numpy as np
import tomllib
from tensorflow.keras.models import model_from_json

with open("config.toml", "rb") as f:
	config = tomllib.load(f)

model_split_into_files = config["model_split_into_files"]
keras_file_path = config["keras_file_path"]

def split_keras_file_into_files():
	os.makedirs(model_split_into_files, exist_ok = True)

	files_to_extract = [
		"config.json",
		"metadata.json",
		"model.weights.h5"
	]

	with zipfile.ZipFile(keras_file_path, "r") as archive:
		for file_name in files_to_extract:
			if file_name in archive.namelist():
				archive.extract(file_name, model_split_into_files)
				print(f"{file_name} extracted")
			else:
				print(f"{file_name} not found")

	print("Extraction complete.")

def convert_h5_model_to_json():
	config_path = Path(model_split_into_files + "/config.json")
	h5_weight_path = Path(model_split_into_files + "/model.weights.h5")
	json_weights_path = Path(model_split_into_files + "/model.weights.json")

	if not config_path.exists():
		raise FileNotFoundError(f"Missing model config file: {config_path}")
	if not h5_weight_path.exists():
		raise FileNotFoundError(f"Missing H5 weights file: {h5_weight_path}")

	with config_path.open("r", encoding = "utf-8") as f:
		config = json.load(f)

	model = model_from_json(json.dumps(config))
	model.load_weights(h5_weight_path)

	serialized = {
		"format": "keras_weights_v1",
		"source": str(h5_weight_path.name),
		"weight_names": [w.name for w in model.weights],
		"weights": [np.asarray(w).tolist() for w in model.get_weights()],
	}

	with json_weights_path.open("w", encoding = "utf-8") as f:
		json.dump(serialized, f)

	print(f"Saved JSON weights to: {json_weights_path}")

## 1 split keras into config.json, metedata.json, model.weights.h5
split_keras_file_into_files()

## 2 convert model.weights.h5 to model.weights.json
convert_h5_model_to_json()

## 3 delete model.weights.h5
os.remove(model_split_into_files + "/model.weights.h5")
