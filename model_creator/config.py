import os

from ganalyzer.model_config import *

# common

latent_dimension_generator = 49  # 49 #121 #225

dataset_name = "humans_fifa"  # "cars_2"
dataset_dimension = str(model_output_size)
dataset_path = os.path.join("datasets", dataset_name, dataset_dimension)

results_root_path = os.path.join("results", dataset_name + "_" + dataset_dimension)
model_path = os.path.join(results_root_path, model_name + "-ls_" + (4 - len(str(latent_dimension_generator))) * "0" + str(latent_dimension_generator))
models_directory = os.path.join(model_path, "models")

models_as_tflite = "models_as_tflite"

# Ensure result folders exist for the active model and the shared plots directory
os.makedirs(models_directory, exist_ok = True)

rgb_images = True

# plotting
PLOTS_DIRECTORY_NAME = "plots"
PLOTS_ROOT_DIRECTORY = os.path.join(results_root_path, PLOTS_DIRECTORY_NAME)
os.makedirs(PLOTS_ROOT_DIRECTORY, exist_ok = True)

RESULTS_DIRECTORY = model_path

statistics_file_path = os.path.join(model_path, "statistics.csv")

# train
batch_size = 32
save_train_epoch_every = 5

# GUI
GUI_tkinter = False
load_quantity_gui = 6

# statistics
all_models = [
	entry
	for entry in sorted(os.listdir(results_root_path))
	if os.path.isdir(os.path.join(results_root_path, entry))
	   and entry != PLOTS_DIRECTORY_NAME
]

every_models_statistics_path = [
	os.path.join(results_root_path, entry) for entry in all_models
]

# comparisons_elements = [('model_0_tiny-ls_0049','epoch_000000'),('model_0_tiny-ls_0049','epoch_000050'),('model_0_tiny-ls_0049','epoch_000100')] #ls 49, trough epochs
# comparisons_elements = [('model_0_tiny-ls_0121','epoch_000000'),('model_0_tiny-ls_0121','epoch_000050'),('model_0_tiny-ls_0121','epoch_000100')] #ls 121, trough epochs
# comparisons_elements = [('model_0_tiny-ls_0225','epoch_000000'),('model_0_tiny-ls_0225','epoch_000050'),('model_0_tiny-ls_0225','epoch_000100')] #ls 225, trough epochs

# comparisons_elements = [('model_0_tiny-ls_0121','epoch_000000'),('model_0_tiny-ls_0121','epoch_000025'),('model_0_tiny-ls_0121','epoch_000050'),('model_0_tiny-ls_0121','epoch_000075'),('model_0_tiny-ls_0121','epoch_000100')] #ls 121, trough epochs
comparisons_elements = [('model_0_tiny-ls_0121', 'epoch_000000'), ('model_0_tiny-ls_0121', 'epoch_000020'), ('model_0_tiny-ls_0121', 'epoch_000040'), ('model_0_tiny-ls_0121', 'epoch_000060'), ('model_0_tiny-ls_0121', 'epoch_000080'), ('model_0_tiny-ls_0121', 'epoch_000100')]  # ls 121, trough epochs

# comparisons_elements = [('model_0_tiny-ls_0049', 'epoch_000100'), ('model_0_tiny-ls_0121', 'epoch_000100'), ('model_0_tiny-ls_0225', 'epoch_000100')]  # ls 121, epoch 100, trough var

nb_comparisons = 200

# Sample outputs
sample_outputs_root_directory = os.path.join(model_path, "sample_outputs")
os.makedirs(sample_outputs_root_directory, exist_ok = True)
