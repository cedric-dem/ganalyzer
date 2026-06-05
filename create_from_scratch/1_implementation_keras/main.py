import json
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.models import model_from_json
import tomllib

with open("config.toml", "rb") as f:
	config = tomllib.load(f)
model_split_into_files = config["model_split_into_files"]

path_to_keras_implementation_intermediary = config["path_to_keras_implementation_intermediary"]


DISPLAY_INSIDE_VALUE = None
SAVE_INTERMEDIARY_VALUES = True

CUSTOM_INPUT_VECTOR = np.array(
	[[
		-0.16, -0.15, 0.19, -0.14, 0.05, 0.66, -0.07, -0.06, -0.04, -0.0, 0.51, -0.05, 0.23, -0.17, -0.06, 0.07, 0.39,
		0.23, -0.08, 0.02, -0.14, -0.16, 0.0, 0.05, 0.09, 0.15, 0.09, -0.12, -0.54, 0.18, -0.44, -0.13, -0.2, 0.03,
		-0.23, -0.14, -0.46, 0.2, 0.03, -0.36, 0.42, -0.12, -0.19, -0.25, -0.23, -0.04, -0.18, 0.18, -0.19
	]],
	dtype = np.float32,
)

ARTIFACTS_DIR = Path(model_split_into_files)
INTERMEDIARY_DIR = Path(path_to_keras_implementation_intermediary)

def save_value(base_dir, index, label, value):
	base_dir.mkdir(parents = True, exist_ok = True)
	safe_label = str(label).replace("/", "_").replace(" ", "_")
	output_path = base_dir / f"values_{index:03d}_{safe_label}.txt"
	array = np.asarray(value)
	output_path.write_text(
		np.array2string(
			array,
			separator = ",",
			threshold = array.size,
			formatter = {"float_kind": lambda x: f"{x:.12f}"},
		),
		encoding = "utf-8",
	)

def save_intermediary_value(index, label, value):
	save_value(INTERMEDIARY_DIR, index, label, value)

def load_model_from_artifacts():
	config_path = ARTIFACTS_DIR / "config.json"
	metadata_path = ARTIFACTS_DIR / "metadata.json"
	weights_path = ARTIFACTS_DIR / "model.weights.json"

	print(f"Loading model artifacts from: {ARTIFACTS_DIR}")

	if not config_path.exists():
		raise FileNotFoundError(f"Missing file: {config_path}")
	if not metadata_path.exists():
		raise FileNotFoundError(f"Missing file: {metadata_path}")
	if not weights_path.exists():
		raise FileNotFoundError(f"Missing file: {weights_path}")

	with config_path.open(encoding = "utf-8") as f:
		config = json.load(f)
	with metadata_path.open(encoding = "utf-8") as f:
		metadata = json.load(f)

	model = model_from_json(json.dumps(config))
	with weights_path.open(encoding = "utf-8") as f:
		weights_payload = json.load(f)

	if "weights" not in weights_payload:
		raise ValueError(f"Invalid JSON weights format: {weights_path}")

	weights = [np.asarray(w, dtype = np.float32) for w in weights_payload["weights"]]
	model.set_weights(weights)
	return model, metadata

def run_with_keras():
	model, _ = load_model_from_artifacts()

	if not model.inputs:
		raise RuntimeError("No model inputs were detected.")

	input_shape = model.inputs[0].shape
	if len(input_shape) != 2 or (input_shape[0] is not None and input_shape[0] <= 0):
		raise ValueError(f"Unsupported input shape: {input_shape}")

	vector_size = int(input_shape[1])
	if CUSTOM_INPUT_VECTOR.shape[1] != vector_size:
		raise ValueError(
			f"Invalid CUSTOM_INPUT_VECTOR size: expected {vector_size}, "
			f"got {CUSTOM_INPUT_VECTOR.shape[1]}"
		)

	batch_input = CUSTOM_INPUT_VECTOR.copy()
	if SAVE_INTERMEDIARY_VALUES:
		save_intermediary_value(0, "original", batch_input)

	if DISPLAY_INSIDE_VALUE is not None:
		layer_index = int(DISPLAY_INSIDE_VALUE)
		layer = model.layers[layer_index]
		print(f"Inspecting layer #{layer_index} ({layer.name})")

		intermediate_model = Model(inputs = model.inputs, outputs = layer.output)
		layer_values = intermediate_model.predict(batch_input, verbose = 0)
		arr = np.asarray(layer_values)
		print(f"Output shape : {arr.shape}")
		print(f"Output min/max : {arr.min():.6f} / {arr.max():.6f}")

	if SAVE_INTERMEDIARY_VALUES:
		intermediary_model = Model(inputs = model.inputs, outputs = [layer.output for layer in model.layers])
		intermediary_outputs = intermediary_model.predict(batch_input, verbose = 0)
		if not isinstance(intermediary_outputs, list):
			intermediary_outputs = [intermediary_outputs]

		for layer_index, layer_output in enumerate(intermediary_outputs, start = 1):
			save_intermediary_value(layer_index, model.layers[layer_index - 1].name, layer_output)

	output = model.predict(batch_input, verbose = 0)
	image = output[0]
	image_uint8 = np.clip((image + 1.0) * 127.5, 0, 255).astype(np.uint8)

	output_path = INTERMEDIARY_DIR / "out.jpg"
	output_path.parent.mkdir(parents = True, exist_ok = True)
	Image.fromarray(image_uint8, mode = "RGB").save(output_path, format = "JPEG", quality = 95)

	print(f"Saved intermediary values to: {INTERMEDIARY_DIR.resolve()}")
	print(f"Saved image to: {output_path}")

if __name__ == "__main__":
	run_with_keras()
