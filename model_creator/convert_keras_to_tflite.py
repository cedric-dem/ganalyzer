from __future__ import annotations

import pathlib
import tensorflow as tf

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
	model_path_tflite.write_bytes(tflite_model)

def _default_models():
	root = pathlib.Path(__file__).resolve().parent
	return (
		(root / "generator.keras", root / "generator.tflite"),
		(root / "discriminator.keras", root / "discriminator.tflite"),
	)

def main():
	for source, target in _default_models():
		print(f"Converting {source.name} -> {target.name}")
		export_tflite(source, target)

if __name__ == "__main__":
	main()
