from config import rgb_images
from save_stats_plot import MODELS_ROOT_PATH, RESULTS_ROOT_PATH
import keras
import numpy as np
import random
import cv2
from keras.preprocessing.image import img_to_array

def apply_model(generator, latent_vector):
	latent_array = np.expand_dims(np.asarray(latent_vector, dtype = "float32"), axis = 0)

	image = generator(latent_array, training = False).numpy()[0]

	if not rgb_images and image.shape[-1] == 3:
		image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		image = np.expand_dims(image, axis = -1)

	new_image = img_to_array(image.astype("float32"))
	output_image = new_image

	output_image = (output_image + 1.0) / 2.0  ##temp fix 1
	output_image = (output_image * 255).clip(0, 255)  ##temp fix 2

	output_image = output_image.astype("uint8")

	output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)  ##temp fix 3

	return output_image

def get_difference_with_original(generator, latent_vector, goal):
	reproduced_image = apply_model(generator, latent_vector)
	# print("shape : ", goal.shape, reproduced_image.shape, type(goal), type(reproduced_image))
	delta = np.abs(goal - reproduced_image)
	return delta.reshape(-1).sum()

def search_random(generator, goal, ls_size, quantity_initial_random):  # returns best latent vector

	current_best_vector = None
	current_best_difference = None

	for current_random_generation in range(quantity_initial_random):
		if (current_random_generation % 100) == 0:
			print("===> Search random, ", current_random_generation, "/", quantity_initial_random)

		new_vector = [get_rnd_elem() for _ in range(ls_size)]  ##todo replace by use of np normal mu 0 sigma 1
		this_difference = get_difference_with_original(generator, new_vector, goal)
		# print("dif", this_difference)
		if current_best_difference is None or current_best_difference > this_difference:
			print("==> new best difference : ", this_difference)
			current_best_vector = new_vector[::]
			current_best_difference = this_difference
		else:
			# print("no improvement difference : ", this_difference, "best is ", current_best_difference)
			pass

	return current_best_vector

def get_rnd_elem():
	return round(random.gauss(0, 1), 2)

def mutate_vector(current_vector, nb_diff):
	new_vector = current_vector[::]

	for _ in range(nb_diff):
		ix = random.randint(0, len(current_vector) - 1)
		new_vector[ix] = get_rnd_elem()

	return new_vector

def search_genetic_algorithm(generator, initial_latent_vector, goal, quantity_genetic_evolution, nb_diff):  # returns best latent vector

	current_best_vector = initial_latent_vector[:]
	current_best_difference = get_difference_with_original(generator, current_best_vector, goal)

	for current_genetic_generation in range(quantity_genetic_evolution):
		if (current_genetic_generation % 100) == 0:
			print("===> Search genetic, ", current_genetic_generation, "/", quantity_genetic_evolution)

		new_vector = mutate_vector(initial_latent_vector, nb_diff)
		this_difference = get_difference_with_original(generator, new_vector, goal)
		# print("dif", this_difference)
		if current_best_difference > this_difference:
			print("==> new best difference : ", this_difference)
			current_best_vector = new_vector[::]
			current_best_difference = this_difference
		else:
			# print("no improvement difference : ", this_difference, "best is ", current_best_difference)
			pass

	return current_best_vector

def save_produced_result(generator, latent_vector, output_path):
	result = apply_model(generator, latent_vector)

	cv2.imwrite(str(output_path), result)

def main_search(generator_name, quantity_initial_random, quantity_genetic_evolution, nb_difference_genetic_algo):
	# open generator
	gen_epoch = 300
	epoch_number = int(str(gen_epoch).replace("epoch_", ""))
	generator_path = MODELS_ROOT_PATH / generator_name / "models" / f"generator_epoch_{epoch_number:06d}.keras"
	output_dir = RESULTS_ROOT_PATH / "imitation"
	generator = keras.models.load_model(generator_path)
	ls_size = int(generator_name.split("_")[-1])

	# open goal image
	goal_image_path = output_dir / "goal_image.png"
	goal = keras.utils.img_to_array(keras.utils.load_img(goal_image_path))

	best_latent_vector = search_random(generator, goal, ls_size, quantity_initial_random)

	best_latent_vector = search_genetic_algorithm(generator, best_latent_vector, goal, quantity_genetic_evolution, nb_difference_genetic_algo)

	# produce best image and save it
	save_produced_result(generator, best_latent_vector, output_dir / "reproduced_image.png")

	print('==> Result : ', best_latent_vector)
	print("==> Total diff : ", get_difference_with_original(generator, best_latent_vector, goal))

main_search("model_1_small_with_h-ls_0121", 1000, 1000, 3)