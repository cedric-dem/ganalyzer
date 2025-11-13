from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import tensorflow as tf

from config import models_as_tflite, models_directory
from ganalyzer.misc import get_list_of_keras_models

def _load_model(model_path):
	return tf.keras.models.load_model(str(model_path), compile = False)

def _build_concrete_function(model):
	input_specs = [tf.TensorSpec(shape = [dim if dim is not None else 1 for dim in tensor.shape], dtype = tensor.dtype) for tensor in model.inputs]

	@tf.function
	def model_fn(*args):
		return model(*args, training = False)

	return model_fn.get_concrete_function(*input_specs)

def _configure_converter(concrete_fn, model):
	converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], model)
	converter.experimental_new_converter = False
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
	converter.allow_custom_ops = False
	converter.optimizations = []
	return converter

def export_tflite(model_path_keras, model_path_tflite):
	model = _load_model(model_path_keras)
	concrete_function = _build_concrete_function(model)
	converter = _configure_converter(concrete_function, model)
	tflite_model = converter.convert()

	target_path = Path(model_path_tflite)
	target_path.parent.mkdir(parents = True, exist_ok = True)
	target_path.write_bytes(tflite_model)

def _default_models():
	last_generator = None
	last_discriminator = None

	for model_name in get_list_of_keras_models():
		model_path = Path(models_directory) / model_name
		if model_name.startswith("discriminator"):
			last_discriminator = model_path
		elif model_name.startswith("generator"):
			last_generator = model_path

	results = []
	if last_generator is not None:
		results.append((last_generator, Path(models_as_tflite) / "generator.tflite"))
	if last_discriminator is not None:
		results.append((last_discriminator, Path(models_as_tflite) / "discriminator.tflite"))
	return results

def main():
	for source, target in _default_models():
		print(f"Converting {source} -> {target}")
		export_tflite(source, target)

if __name__ == "__main__":
	main()
