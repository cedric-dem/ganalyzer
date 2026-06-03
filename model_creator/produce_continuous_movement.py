from ganalyzer.misc import get_last_epoch_available
from pathlib import Path

import keras
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from config import LATENT_DIMENSION_GENERATOR, MODEL_NAME, IS_RGB_IMAGES, MODELS_DIRECTORY, CONTINUOUS_MOVEMENT_DIRECTORY, CONTINUOUS_MOVEMENT_LENGTH, CONTINUOUS_MOVEMENT_NUMBER_CHANGES, GENERATOR_GLOBAL_NAME, EPOCH_GLOBAL_NAME, CONTINUOUS_MOVEMENT_IMAGE_PREFIX

def get_fake_images_sample(generator_name, length_evolution, nb_changes):
	models_dir = Path(MODELS_DIRECTORY)

	gen_epoch = get_last_epoch_available(GENERATOR_GLOBAL_NAME, str(models_dir))
	print('Generating fake images using ', generator_name, gen_epoch)

	generator_path = models_dir / str(GENERATOR_GLOBAL_NAME + "_" + EPOCH_GLOBAL_NAME + f"_{gen_epoch:06d}.keras")
	output_dir = Path(CONTINUOUS_MOVEMENT_DIRECTORY)
	output_dir.mkdir(parents = True, exist_ok = True)

	generator = keras.models.load_model(generator_path)
	ls_size = LATENT_DIMENSION_GENERATOR

	latent_vector = np.random.normal(0.0, 1.0, size = (1, ls_size))

	for i in range(length_evolution):
		print("=> Generating image ", i + 1, "/", length_evolution)
		print(latent_vector.shape)

		indices = np.random.choice(latent_vector.shape[1], size = nb_changes, replace = False)
		new_values = np.random.normal(0.0, 1.0, size = nb_changes)
		latent_vector[0, indices] = new_values

		image = generator(latent_vector, training = False).numpy()[0]

		if not IS_RGB_IMAGES and image.shape[-1] == 3:
			image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			image = np.expand_dims(image, axis = -1)

		new_image = img_to_array(image.astype("float32"))
		output_image = new_image

		output_image = (output_image + 1.0) / 2.0  ##temp fix 1
		output_image = (output_image * 255).clip(0, 255)  ##temp fix 2

		output_image = output_image.astype("uint8")

		output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)  ##temp fix 3

		output_path = output_dir / str(CONTINUOUS_MOVEMENT_IMAGE_PREFIX + f"_{i + 1:04d}.png")
		cv2.imwrite(str(output_path), output_image)

generator_name = f"{MODEL_NAME}-ls_{LATENT_DIMENSION_GENERATOR:04d}"
get_fake_images_sample(generator_name, CONTINUOUS_MOVEMENT_LENGTH, CONTINUOUS_MOVEMENT_NUMBER_CHANGES)
