from save_stats_plot import RESULTS_ROOT_PATH
import keras
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from config import rgb_images

def get_fake_images_sample(generator_name, length_evolution, nb_changes):
	gen_epoch = 200
	print('Generating fake images using ', generator_name, gen_epoch)
	epoch_number = int(str(gen_epoch).replace("epoch_", ""))

	generator_path = RESULTS_ROOT_PATH / generator_name / "models" / f"generator_epoch_{epoch_number:06d}.keras"
	output_dir = RESULTS_ROOT_PATH.parent / "evolution_sample"
	output_dir.mkdir(parents = True, exist_ok = True)

	generator = keras.models.load_model(generator_path)
	ls_size = int(generator_name.split("_")[-1])

	latent_vector = np.random.normal(0.0, 1.0, size = (1, ls_size))

	for i in range(length_evolution):
		print("=> Generating image ", i + 1, "/", length_evolution)
		print(latent_vector.shape)

		indices = np.random.choice(latent_vector.shape[1], size = nb_changes, replace = False)
		new_values = np.random.normal(0.0, 1.0, size = nb_changes)
		latent_vector[0, indices] = new_values

		image = generator(latent_vector, training = False).numpy()[0]

		if not rgb_images and image.shape[-1] == 3:
			image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			image = np.expand_dims(image, axis = -1)

		new_image = img_to_array(image.astype("float32"))
		output_image = new_image

		output_image = (output_image + 1.0) / 2.0  ##temp fix 1
		output_image = (output_image * 255).clip(0, 255)  ##temp fix 2

		output_image = output_image.astype("uint8")

		output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)  ##temp fix 3

		output_path = output_dir / f"evo_{i + 1:04d}.jpg"
		cv2.imwrite(str(output_path), output_image)

length_evolution = 100
nb_changes = 10
get_fake_images_sample("model_0_small-ls_0225", length_evolution, nb_changes)
