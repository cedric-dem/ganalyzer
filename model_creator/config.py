import os

from ganalyzer.model_config import *

# common
additional_dense_units = 500

latent_dimension_generator_available = [49, 121, 225, 400]  # maybe 49, 100, 196, 400
latent_dimension_generator = latent_dimension_generator_available[0]

dataset_name = "humans_fifa"  # "cars_2"
dataset_dimension = str(model_output_size)
dataset_path = os.path.join("datasets", dataset_name, dataset_dimension)

results_root_path = os.path.join("results", dataset_name)
models_root_path = os.path.join(results_root_path, "models")
model_path = os.path.join(models_root_path, model_name + "-ls_" + (4 - len(str(latent_dimension_generator))) * "0" + str(latent_dimension_generator)) #todo isolate in function str_ls
models_directory = os.path.join(model_path, "models")

models_as_tflite = os.path.join(results_root_path, "models_as_tf_lite")

# Ensure result folders exist for the active model and the shared plots directory
os.makedirs(models_root_path, exist_ok = True)
os.makedirs(models_directory, exist_ok = True)
os.makedirs(models_as_tflite, exist_ok = True)

rgb_images = True

# plotting
PLOTS_DIRECTORY_NAME = "plots"
PLOTS_ROOT_DIRECTORY = os.path.join(results_root_path, PLOTS_DIRECTORY_NAME)
os.makedirs(PLOTS_ROOT_DIRECTORY, exist_ok = True)

IMITATION_ROOT_DIRECTORY = os.path.join(results_root_path, "imitation")
os.makedirs(IMITATION_ROOT_DIRECTORY, exist_ok = True)
EVOLUTION_SAMPLE_ROOT_DIRECTORY = os.path.join(results_root_path, "evolution_sample")
os.makedirs(EVOLUTION_SAMPLE_ROOT_DIRECTORY, exist_ok = True)

PLOTS_HEATMAP_EPOCHS_DIRECTORY = os.path.join(PLOTS_ROOT_DIRECTORY, "heatmap_epochs")
os.makedirs(PLOTS_HEATMAP_EPOCHS_DIRECTORY, exist_ok = True)
PLOTS_HEATMAP_MODEL_SIZE_DIRECTORY = os.path.join(PLOTS_ROOT_DIRECTORY, "heatmap_model_size")
os.makedirs(PLOTS_HEATMAP_MODEL_SIZE_DIRECTORY, exist_ok = True)
PLOTS_HEATMAP_LATENT_SPACE_SIZE_DIRECTORY = os.path.join(PLOTS_ROOT_DIRECTORY, "heatmap_latent_space_size")
os.makedirs(PLOTS_HEATMAP_LATENT_SPACE_SIZE_DIRECTORY, exist_ok = True)
PLOTS_NUMBER_PARAMETERS_DIRECTORY = os.path.join(PLOTS_ROOT_DIRECTORY, "number_parameters")
os.makedirs(PLOTS_NUMBER_PARAMETERS_DIRECTORY, exist_ok = True)

PATH_LOSS_PLOTS = os.path.join(PLOTS_ROOT_DIRECTORY, "loss")
os.makedirs(PATH_LOSS_PLOTS, exist_ok = True)

PATH_LOSS_BY_LS_PLOTS = os.path.join(PATH_LOSS_PLOTS, "by_ls_size")
os.makedirs(PATH_LOSS_BY_LS_PLOTS, exist_ok = True)
PATH_LOSS_BY_MODEL_PLOTS =  os.path.join(PATH_LOSS_PLOTS, "by_model_name")
os.makedirs(PATH_LOSS_BY_MODEL_PLOTS, exist_ok = True)

RESULTS_DIRECTORY = model_path

statistics_file_path = os.path.join(model_path, "statistics.csv")

# train
batch_size = 32
save_train_epoch_every = 5

# GUI
show_inside_values = True
GUI_tkinter = False
load_quantity_gui = 3#6

# statistics
all_models = [
	entry
	for entry in sorted(os.listdir(models_root_path))
	if os.path.isdir(os.path.join(models_root_path, entry))
]

every_models_statistics_path = [
	os.path.join(models_root_path, entry) for entry in all_models
]

nb_epoch_taken_comparison = 5
nb_comparisons = 100

# Sample outputs
sample_outputs_root_directory = os.path.join(model_path, "sample_outputs")
os.makedirs(sample_outputs_root_directory, exist_ok = True)