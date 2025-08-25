import os

#common
dataset_name = "humans_fifa" #"cars_2"
dataset_dimension="64"
dataset_path = "datasets/"+dataset_name+"/"+dataset_dimension

model_name = "model_c"
global_path = "models/" + dataset_name+"/"+dataset_dimension + "/"
model_path = global_path + model_name + "/"

os.makedirs(model_path, exist_ok=True)

rgb_images = True
latent_dimension_generator = 121

statistics_file_path = model_path + "statistics.csv"

#train
batch_size = 32
save_train_epoch_every = 1

#GUI
load_quantity_gui = 15

#statistics
show_every_models_statistic = True
all_models = os.listdir(global_path)
every_models_statistics_path = [global_path + i for i in all_models]