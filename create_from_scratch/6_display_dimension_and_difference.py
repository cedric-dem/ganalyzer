from __future__ import annotations

import numpy as np
import ast
from pathlib import Path

import tomllib

with open("config.toml", "rb") as f:
	config = tomllib.load(f)

path_to_keras_implementation_intermediary = config["path_to_keras_implementation_intermediary"]
path_to_non_keras_implementation_intermediary = config["path_to_non_keras_implementation_intermediary"]
path_to_cpp_implementation_intermediary = config["path_to_cpp_implementation_intermediary"]
path_to_rust_implementation_intermediary = config["path_to_rust_implementation_intermediary"]
path_to_c_implementation_intermediary = config["path_to_c_implementation_intermediary"]

DEFAULT_KERAS_DIR = Path(path_to_keras_implementation_intermediary)
DEFAULT_NON_KERAS_DIR = Path(path_to_non_keras_implementation_intermediary)
DEFAULT_CPP_DIR = Path(path_to_cpp_implementation_intermediary)
DEFAULT_RUST_DIR = Path(path_to_rust_implementation_intermediary)
DEFAULT_C_DIR = Path(path_to_c_implementation_intermediary)

def infer_shape(value):
	if isinstance(value, (list, tuple)):
		length = len(value)
		if length == 0:
			return (0,)

		child_shapes = [infer_shape(item) for item in value]
		first_shape = child_shapes[0]
		if any(shape != first_shape for shape in child_shapes[1:]):
			raise ValueError("jagged/non-rectangular array detected")

		return (length, *first_shape)

	return ()

def shape_to_text(shape):
	if not shape:
		return "scalar"
	return "x".join(str(dim) for dim in shape)

def load_array_shape(file_path):
	raw = file_path.read_text(encoding = "utf-8").strip()
	if not raw:
		raise ValueError("file is empty")

	parsed = ast.literal_eval(raw)
	return infer_shape(parsed)

def describe_shape(file_path):
	if not file_path.exists():
		return "ERROR (missing file)"

	try:
		return shape_to_text(load_array_shape(file_path))
	except Exception as exc:  # keep processing remaining files
		return f"ERROR ({exc})"

def list_text_files(folder):
	files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() != ".jpg"]
	return sorted(files, key = lambda p: p.name)

def display_dimension():
	is_all_the_same_dimension = True

	if not DEFAULT_KERAS_DIR.is_dir():
		print(f"Keras intermediary folder does not exist: {DEFAULT_KERAS_DIR}")
		return

	keras_files = list(list_text_files(DEFAULT_KERAS_DIR))
	if not keras_files:
		print(f"No eligible files found in {DEFAULT_KERAS_DIR}")
		return

	for keras_file in keras_files:
		keras_shape = describe_shape(keras_file)
		non_keras_shape = describe_shape(DEFAULT_NON_KERAS_DIR / keras_file.name)
		c_shape = describe_shape(DEFAULT_CPP_DIR / keras_file.name)
		cpp_shape = describe_shape(DEFAULT_CPP_DIR / keras_file.name)
		rust_shape = describe_shape(DEFAULT_RUST_DIR / keras_file.name)

		current_shapes = (keras_shape, non_keras_shape, c_shape, cpp_shape, rust_shape)
		current_is_same_dimension = (
				len(set(current_shapes)) == 1
				and not any(shape.startswith("ERROR") for shape in current_shapes)
		)
		is_all_the_same_dimension = is_all_the_same_dimension and current_is_same_dimension

		print(
			f"{keras_file.stem} : "
			f"(1)keras : {keras_shape} "
			f"(2)non keras : {non_keras_shape} "
			f"(3)c : {c_shape} "
			f"(4)c++ : {cpp_shape} "
			f"(5)rust : {rust_shape}"
		)

	print(f"is_all_the_same_dimension : {is_all_the_same_dimension}")
	return is_all_the_same_dimension

def load_array(file_path):
	raw = file_path.read_text(encoding = "utf-8").strip()
	if not raw:
		raise ValueError("file is empty")

	parsed = ast.literal_eval(raw)
	return np.asarray(parsed, dtype = np.float64)

def list_text_files(folder):
	files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() != ".jpg"]
	return sorted(files, key = lambda p: p.name)

def describe_difference(reference_file, candidate_file):
	if not candidate_file.exists():
		return "ERROR (missing file)"

	try:
		reference_values = load_array(reference_file)
		candidate_values = load_array(candidate_file)

		if reference_values.shape != candidate_values.shape:
			return (
				"ERROR (shape mismatch "
				f"{reference_values.shape} vs {candidate_values.shape})"
			)

		differences = np.abs(reference_values - candidate_values).reshape(-1)
		return (
			f"min={np.min(differences):.12f} "
			f"max={np.max(differences):.12f} "
			f"avg={np.mean(differences):.12f} "
			f"median={np.median(differences):.12f}"
		)
	except Exception as exc:  # keep processing remaining files
		return f"ERROR ({exc})"

def display_difference():
	if not DEFAULT_KERAS_DIR.is_dir():
		print(f"Keras intermediary folder does not exist: {DEFAULT_KERAS_DIR}")
		return

	keras_files = list(list_text_files(DEFAULT_KERAS_DIR))
	if not keras_files:
		print(f"No eligible files found in {DEFAULT_KERAS_DIR}")
		return

	for keras_file in keras_files:
		non_keras_difference = describe_difference(keras_file, DEFAULT_NON_KERAS_DIR / keras_file.name)
		c_difference = describe_difference(keras_file, DEFAULT_CPP_DIR / keras_file.name)
		cpp_difference = describe_difference(keras_file, DEFAULT_CPP_DIR / keras_file.name)
		rust_difference = describe_difference(keras_file, DEFAULT_RUST_DIR / keras_file.name)

		print(
			f"\n{keras_file.stem} : "
			f"\n   (1)keras vs (2)non keras : {non_keras_difference} "
			f"\n   (1)keras vs (3)c : {c_difference}"
			f"\n   (1)keras vs (4)c++ : {cpp_difference}"
			f"\n   (1)keras vs (5)rust : {rust_difference}"
		)

if __name__ == "__main__":
	if display_dimension():
		display_difference()
