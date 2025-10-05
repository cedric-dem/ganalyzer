import os

from ganalyzer.model_config import *

# common
dataset_name = "humans_fifa"  # "cars_2"
dataset_dimension = str(model_output_size)
dataset_path = os.path.join("datasets", dataset_name, dataset_dimension)

results_root_path = os.path.join("results", dataset_name, dataset_dimension)
model_path = os.path.join(results_root_path, model_name)
models_directory = os.path.join(model_path, "models")

os.makedirs(models_directory, exist_ok=True)

rgb_images = True
latent_dimension_generator = 121

statistics_file_path = os.path.join(model_path, "statistics.csv")

#train
batch_size = 32
save_train_epoch_every = 1

#GUI
load_quantity_gui = 9

#statistics
show_every_models_statistic = True
all_models = [
    entry
    for entry in sorted(os.listdir(results_root_path))
    if os.path.isdir(os.path.join(results_root_path, entry))
]
every_models_statistics_path = [os.path.join(results_root_path, entry) for entry in all_models]