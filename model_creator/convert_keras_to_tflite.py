from __future__ import annotations

from pathlib import Path
import tensorflow as tf

from config import model_path, models_directory, models_as_tflite
from ganalyzer.misc import get_list_of_keras_models

def _load_model(model_path):
	return tf.keras.models.load_model(str(model_path), compile = False)

def _build_concrete_function(model):
	input_specs = []
	for tensor in model.inputs:
		shape = [dim if dim is not None else 1 for dim in tensor.shape]
		input_specs.append(tf.TensorSpec(shape = shape, dtype = tensor.dtype))

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
	model_path_tflite = Path(model_path_tflite)
	model_path_tflite.write_bytes(tflite_model)

def _default_models():
	all_models = get_list_of_keras_models()
	all_models.sort()

	last_discriminator = None
	last_generator = None

	for current_model in all_models:
		if current_model.startswith('discriminator'):
			last_discriminator = current_model
		elif current_model.startswith('generator'):
			last_generator = current_model

	return ((models_directory + "/" + last_generator, models_as_tflite + "/generator.tflite"),
			(models_directory + "/" + last_discriminator, models_as_tflite + "/discriminator.tflite"))

def main():
	for source, target in _default_models():
		print(f"Converting {source} -> {target}")
		export_tflite(source, target)

if __name__ == "__main__":
	main()
