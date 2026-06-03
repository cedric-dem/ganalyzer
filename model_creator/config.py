model_output_size = 100

all_models = ["model_0_small", "model_1_medium", "model_2_large"]
model_name = all_models[1]

# common
latent_dimension_generator_available = [49, 100, 196]
latent_dimension_generator = latent_dimension_generator_available[1]

rgb_images = True

# train
batch_size = 32
save_train_epoch_every = 10

# GUI
load_quantity_gui = 3  # 6

# statistics
nb_epoch_taken_comparison = 5
nb_comparisons = 100

# PATH
model_name_explicit = f"{model_name}-ls_{latent_dimension_generator:04d}"

dataset_name = "humans_fifa"  # "cars_2"
dataset_dimension = str(model_output_size)
dataset_path = "datasets/" + dataset_name + "/" + dataset_dimension

results_root_path = "results/" + dataset_name
models_root_path = results_root_path + "/models"
model_path = models_root_path + "/" + model_name_explicit
models_directory_name = "models_files"
models_directory = model_path + "/" + models_directory_name

models_as_tflite = model_path + "/models_as_tf_lite"

DISCRIMINATOR_GLOBAL_NAME = "discriminator"
GENERATOR_GLOBAL_NAME = "generator"
EPOCH_GLOBAL_NAME = "epoch"
LATENT_SPACE_GLOBAL_NAME = "ls"

EVOLUTION_IMG_PREFIX = "evo"

# plot
PLOTS_ROOT_DIRECTORY = results_root_path + "/plots"

PLOTS_HEATMAP_EPOCHS_DIRECTORY = PLOTS_ROOT_DIRECTORY + "/heatmap_epochs"
PLOTS_HEATMAP_MODEL_SIZE_DIRECTORY = PLOTS_ROOT_DIRECTORY + "/heatmap_model_size"
PLOTS_HEATMAP_LATENT_SPACE_SIZE_DIRECTORY = PLOTS_ROOT_DIRECTORY + "/heatmap_latent_space_size"
PLOTS_NUMBER_PARAMETERS_DIRECTORY = PLOTS_ROOT_DIRECTORY + "/number_parameters"

PATH_LOSS_PLOTS = PLOTS_ROOT_DIRECTORY + "/loss"

PATH_LOSS_BY_LS_PLOTS = PATH_LOSS_PLOTS + "/by_ls_size"
PATH_LOSS_BY_MODEL_PLOTS = PATH_LOSS_PLOTS + "/by_model_name"

STATISTICS_CSV_FILENAME = "statistics.csv"
statistics_file_path = model_path + "/" + STATISTICS_CSV_FILENAME

# Sample outputs
sample_outputs_root_directory = model_path + "/sample_outputs"

evolution_sample_dir = model_path + "/evolution_sample"

# evolution
evolution_length = 100
evolution_number_changes = 2

# reproduction
reproduced_images_output_dir = model_path + "/reproduced_images"
IMAGE_TO_REPRODUCE = dataset_path + "/2.png"

reproduced_image_prefix = "reproduced_image"
QTY_INITIAL_RANDOM = 10
QTY_GENETIC_EVO = 10
QTY_GENETIC_ALGO = 1
NB_RETRIES_AVG = 10

# Plot names
plot_names = {
	"every_generator_loss": "Generator Loss Over Epochs",
	"every_discriminator_loss": "Discriminator Loss Over Epochs",
	"split_generator_loss": "Generator Loss Over Epochs for MODEL_NAME",
	"split_discriminator_loss": "Discriminator Loss Over Epochs for MODEL_NAME",
	"current_number_epoch": "Number of training epochs",
	"number_parameters": "Number of trainable parameters for MODEL_NAME",
	"median_time_per_epoch": "Time per epoch, in seconds",
	"comparison_heatmap": "Heatmap for  MODEL_NAME",
	"latent_space_size_comparison_heatmap": "Heatmap for  ls_size =  LATENT_SPACE_SIZE",
	"model_size_comparison_heatmap": "Heatmap for  model_size MODEL_NAME",
}

plot_image_names = {
	"every_generator_loss": "every_generator_loss",
	"every_discriminator_loss": "every_discriminator_loss",
	"split_generator_loss": "MODEL_NAME_generator_loss",
	"split_discriminator_loss": "MODEL_NAME_discriminator_loss",
	"current_number_epoch": "current_number_epochs",
	"number_parameters": "parameters_per_model_MODEL_NAME",
	"median_time_per_epoch": "time_per_epoch",
	"comparison_heatmap": "MODEL_NAME",
	"latent_space_size_comparison_heatmap": "ls_size =  LATENT_SPACE_SIZE",
	"model_size_comparison_heatmap": "model_size MODEL_NAME",
}

x_label_names = {
	"heatmap": "Latent Space Size",
	"loss": "Epoch",
	"comparison_heatmap": "Discriminator",
}

y_label_names = {
	"heatmap": "Model Size",
	"loss": "Loss",
	"comparison_heatmap": "Generator",
}
