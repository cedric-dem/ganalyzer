from pathlib import Path

import cv2
import keras
import numpy as np
from keras.preprocessing.image import img_to_array

from config import (
	EPOCH_GLOBAL_NAME,
	EVOLUTION_SAMPLE_PATH,
	EVOLUTION_SAMPLE_PREFIX,
	GENERATOR_GLOBAL_NAME,
	IS_RGB_IMAGES,
	LATENT_DIMENSION_GENERATOR,
	MODEL_NAME,
	MODELS_DIRECTORY,
)
from ganalyzer.misc import get_list_of_keras_models

def _get_available_generator_paths():
	models_dir = Path(MODELS_DIRECTORY)
	generator_prefix = f"{GENERATOR_GLOBAL_NAME}_{EPOCH_GLOBAL_NAME}_"

	return [
		models_dir / model_name
		for model_name in get_list_of_keras_models(str(models_dir))
		if model_name.startswith(generator_prefix) and model_name.endswith(".keras")
	]

def _prepare_output_image(image):
	if not IS_RGB_IMAGES and image.shape[-1] == 3:
		image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		image = np.expand_dims(image, axis = -1)

	output_image = img_to_array(image.astype("float32"))
	output_image = (output_image + 1.0) / 2.0
	output_image = (output_image * 255).clip(0, 255)
	output_image = output_image.astype("uint8")

	if output_image.shape[-1] == 3:
		output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

	return output_image

def get_fake_images_sample(generator_name):
	generator_paths = _get_available_generator_paths()
	if not generator_paths:
		raise ValueError(f"No generator models available in {MODELS_DIRECTORY}.")

	print(f"Generating evolution sample for {generator_name} using {len(generator_paths)} generators")

	output_dir = Path(EVOLUTION_SAMPLE_PATH)
	output_dir.mkdir(parents = True, exist_ok = True)

	latent_vector = np.random.normal(0.0, 1.0, size = (1, LATENT_DIMENSION_GENERATOR))

	for index, generator_path in enumerate(generator_paths, start = 1):
		print(f"=> Generating image {index}/{len(generator_paths)} with {generator_path.name}")

		generator = keras.models.load_model(generator_path)
		image = generator(latent_vector, training = False).numpy()[0]
		output_image = _prepare_output_image(image)

		epoch = generator_path.stem.split("_")[-1]
		output_path = output_dir / f"{EVOLUTION_SAMPLE_PREFIX}_{epoch}.png"
		cv2.imwrite(str(output_path), output_image)

if __name__ == "__main__":
	generator_name = f"{MODEL_NAME}-ls_{LATENT_DIMENSION_GENERATOR:04d}"
	get_fake_images_sample(generator_name)
