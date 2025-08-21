import os

dataset_name = "humans_1"
dataset_dimension="64"

##############################

alpha = 0.15

model_name = "model_a"

model_path = "models/" + dataset_name+"/"+dataset_dimension + "/" +model_name + "/"

os.makedirs(model_path, exist_ok=True)

statistics_file_path = model_path + "statistics.csv"

rgb_images = True

latent_dimension_generator = 121

batch_size = 32

save_train_epoch_every = 1

load_quantity_gui = 15

dataset_path = "datasets/"+dataset_name+"/"+dataset_dimension
