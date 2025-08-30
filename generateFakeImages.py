from misc import *
from ModelViewer import *
import numpy as np
import random
from PIL import Image

def saveImage(arr, name):
	arr_uint8 = arr.astype(np.uint8)
	img = Image.fromarray(arr_uint8, 'RGB')
	img.save(name)

def generate_image_from_input_values(input_raw, current_model, image_path):

	current_input = np.array([input_raw])

	if rgb_images:
		predicted_raw = current_model.predict(current_input)[0, :, :, :]
	else:
		predicted_raw = current_model.predict(current_input)[0, :, :, 0]

	# return find_limits_and_project(predicted_raw)
	res = find_limits_and_project(predicted_raw)
	#print('test', type(res), res.shape, np.max(res), np.min(res))
	saveImage(res, image_path)

def load_model(index_chosen):
	available_epochs = get_available_epochs()
	this_filename = get_model_path_at_given_epoch_closest_possible("generator", index_chosen, available_epochs)
	print(f"=> will load generator epoch {index_chosen}, closest found is : {this_filename}")
	current_model = keras.models.load_model(this_filename)
	return current_model

img_quantity=100

for index_chosen in [0,50,100,150,200,250,300]:

	current_model = load_model(index_chosen)

	path="fake_images/"+model_name+"/"+str(index_chosen)+"/"

	os.makedirs(path, exist_ok=True)

	for i in range(img_quantity):
		#could make 1 batch with size img_quantity but meh
		noise  = np.random.normal(0, 1, (1, latent_dimension_generator))[0]
		#noise = [random.random() for _ in range(121)]

		image_path = path+"fake_"+str(i)+".jpg"
		generate_image_from_input_values(noise, current_model, image_path)