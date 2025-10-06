import os

from ganalyzer.model_config import *

# common
dataset_name = "humans_fifa"  # "cars_2"
dataset_dimension = str(model_output_size)
dataset_path = os.path.join("datasets", dataset_name, dataset_dimension)

results_root_path = os.path.join("results", dataset_name, dataset_dimension)
model_path = os.path.join(results_root_path, model_name)
models_directory = os.path.join(model_path, "models")

# Ensure result folders exist for the active model and the shared plots directory
os.makedirs(models_directory, exist_ok=True)

rgb_images = True
latent_dimension_generator = 121

# plotting
PLOTS_DIRECTORY_NAME = "plots"
PLOTS_ROOT_DIRECTORY = os.path.join(results_root_path, PLOTS_DIRECTORY_NAME)
MODEL_PLOTS_DIRECTORY = os.path.join(PLOTS_ROOT_DIRECTORY, model_name)
os.makedirs(MODEL_PLOTS_DIRECTORY, exist_ok=True)

RESULTS_DIRECTORY = model_path

statistics_file_path = os.path.join(model_path, "statistics.csv")

#train
batch_size = 30
save_train_epoch_every = 10

#GUI
load_quantity_gui = 9

#statistics
show_every_models_statistic = True
if show_every_models_statistic:
    all_models = [
        entry
        for entry in sorted(os.listdir(results_root_path))
        if os.path.isdir(os.path.join(results_root_path, entry))
        and entry != PLOTS_DIRECTORY_NAME
    ]
else:
    all_models = [model_name]

every_models_statistics_path = [
    os.path.join(results_root_path, entry) for entry in all_models
]