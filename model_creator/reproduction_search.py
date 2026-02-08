from config import latent_dimension_generator, model_name, rgb_images
from save_stats_plot import MODELS_ROOT_PATH, RESULTS_ROOT_PATH
import keras
import numpy as np
import random
import cv2
import copy
from ganalyzer.misc import get_last_epoch_available
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
			current_best_vector = copy.deepcopy(new_vector)
			current_best_difference = this_difference
		else:
			# print("no improvement difference : ", this_difference, "best is ", current_best_difference)
			pass

	return current_best_vector

def get_rnd_elem():
	return round(random.gauss(0, 1), 2)

def mutate_vector(current_vector, nb_diff):
	new_vector = copy.deepcopy(current_vector)  # could be optimized, deepcopy only when found new

	for _ in range(nb_diff):
		ix = random.randint(0, len(current_vector) - 1)
		new_vector[ix] = get_rnd_elem()

	return new_vector

def search_genetic_algorithm(generator, initial_latent_vector, goal, quantity_genetic_evolution, nb_diff):  # returns best latent vector

	current_best_vector = copy.deepcopy(initial_latent_vector)
	current_best_difference = get_difference_with_original(generator, current_best_vector, goal)

	for current_genetic_generation in range(quantity_genetic_evolution):
		if (current_genetic_generation % 100) == 0:
			print("===> Search genetic, ", current_genetic_generation, "/", quantity_genetic_evolution)

		new_vector = mutate_vector(initial_latent_vector, nb_diff)
		this_difference = get_difference_with_original(generator, new_vector, goal)
		# print("dif", this_difference)
		if current_best_difference > this_difference:
			print("==> new best difference : ", this_difference)
			current_best_vector = copy.deepcopy(new_vector)
			current_best_difference = this_difference
		else:
			# print("no improvement difference : ", this_difference, "best is ", current_best_difference)
			pass

	return current_best_vector

def save_produced_result(generator, latent_vector, output_path):
	result = apply_model(generator, latent_vector)

	cv2.imwrite(str(output_path), result)

def main_search(generator_name, quantity_initial_random, quantity_genetic_evolution, nb_difference_genetic_algo, nb_retries_avg):
	# open generator
	models_dir = MODELS_ROOT_PATH / generator_name / "models"
	gen_epoch = get_last_epoch_available("generator", str(models_dir))
	generator_path = models_dir / f"generator_epoch_{gen_epoch:06d}.keras"

	output_dir = RESULTS_ROOT_PATH / "imitation"
	generator = keras.models.load_model(generator_path)
	ls_size = latent_dimension_generator

	# open goal image
	goal_image_path = output_dir / "goal_image.png"
	goal = keras.utils.img_to_array(keras.utils.load_img(goal_image_path))

	all_best_latent_vectors = []

	for current_avg in range(nb_retries_avg):
		print("========> NEW ,", current_avg)
		best_latent_vector = search_random(generator, goal, ls_size, quantity_initial_random)

		best_latent_vector = search_genetic_algorithm(generator, best_latent_vector, goal, quantity_genetic_evolution, nb_difference_genetic_algo)

		# produce best image and save it
		save_produced_result(generator, best_latent_vector, output_dir / str("zz_reproduced_image_" + str(current_avg) + ".png"))

		all_best_latent_vectors.append(best_latent_vector)
		print('==> finihed, this Result : ', best_latent_vector)

	overall_avg_latent_vector = get_avg_latent_vector(all_best_latent_vectors)

	save_produced_result(generator, overall_avg_latent_vector, output_dir / "reproduced_image.png")

	print('==> Result : ', overall_avg_latent_vector)
	print("==> Total diff : ", get_difference_with_original(generator, overall_avg_latent_vector, goal))

def get_avg_latent_vector(all_best_latent_vectors):
	total = [0 for _ in range(len(all_best_latent_vectors[0]))]

	for current_b in all_best_latent_vectors:
		for i in range(len(current_b)):
			total[i] += current_b[i]

	for i in range(len(total)):
		total[i] /= len(all_best_latent_vectors)

	return total

generator_name = f"{model_name}-ls_{latent_dimension_generator:04d}"
main_search(generator_name, 10, 10, 1, 10)
