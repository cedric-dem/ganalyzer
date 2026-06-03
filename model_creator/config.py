# General
MODEL_OUTPUT_SIZE = 100

ALL_MODELS = ["model_0_small", "model_1_medium", "model_2_large"]
MODEL_NAME = ALL_MODELS[1]

LATENT_DIMENSION_GENERATOR_AVAILABLE = [49, 100, 196]
LATENT_DIMENSION_GENERATOR = LATENT_DIMENSION_GENERATOR_AVAILABLE[1]

IS_RGB_IMAGES = True

# train
BATCH_SIZE = 32
SAVE_TRAIN_EPOCH_EVERY = 10

# GUI
LOAD_QUANTITY_GUI = 3  # 6

# statistics
NUMBER_EPOCH_TAKEN_COMPARISON = 5
NUMBER_COMPARISON = 100

# NAMING
DISCRIMINATOR_GLOBAL_NAME = "discriminator"
GENERATOR_GLOBAL_NAME = "generator"
EPOCH_GLOBAL_NAME = "epoch"
LATENT_SPACE_GLOBAL_NAME = "ls"

EVOLUTION_IMG_PREFIX = "evo"

# reproduce
QUANTITY_INITIAL_RANDOM = 10
QUANTITY_GENETIC_EVO = 10
QUANTITY_GENETIC_ALGO = 1
NB_RETRIES_AVG = 10

# Plot Names
PLOT_NAMES = {
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

PLOT_IMAGE_NAMES = {
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

X_LABEL_NAMES = {
	"heatmap": "Latent Space Size",
	"loss": "Epoch",
	"comparison_heatmap": "Discriminator",
}

Y_LABEL_NAMES = {
	"heatmap": "Model Size",
	"loss": "Loss",
	"comparison_heatmap": "Generator",
}

# global path
MODEL_NAME_EXPLICIT = f"{MODEL_NAME}-ls_{LATENT_DIMENSION_GENERATOR:04d}"

DATASET_NAME = "humans_fifa"  # "cars_2"
DATASET_DIMENSION = str(MODEL_OUTPUT_SIZE)
DATASET_PATH = "datasets/" + DATASET_NAME + "/" + DATASET_DIMENSION

RESULTS_ROOT_PATH = "results/" + DATASET_NAME
MODELS_ROOT_PATH = RESULTS_ROOT_PATH + "/models"
MODELS_PATH = MODELS_ROOT_PATH + "/" + MODEL_NAME_EXPLICIT
MODELS_DIRECTORY_NAME = "models_files"
MODELS_DIRECTORY = MODELS_PATH + "/" + MODELS_DIRECTORY_NAME

MODELS_AS_TFLITE = MODELS_PATH + "/models_as_tf_lite"

STATISTICS_CSV_FILENAME = "statistics.csv"
STATISTICS_FILE_PATH = MODELS_PATH + "/" + STATISTICS_CSV_FILENAME

# Path plot
PLOTS_ROOT_DIRECTORY = RESULTS_ROOT_PATH + "/plots"

PLOTS_HEATMAP_EPOCHS_DIRECTORY = PLOTS_ROOT_DIRECTORY + "/heatmap_epochs"
PLOTS_HEATMAP_MODEL_SIZE_DIRECTORY = PLOTS_ROOT_DIRECTORY + "/heatmap_model_size"
PLOTS_HEATMAP_LATENT_SPACE_SIZE_DIRECTORY = PLOTS_ROOT_DIRECTORY + "/heatmap_latent_space_size"
PLOTS_NUMBER_PARAMETERS_DIRECTORY = PLOTS_ROOT_DIRECTORY + "/number_parameters"

PATH_LOSS_PLOTS = PLOTS_ROOT_DIRECTORY + "/loss"

PATH_LOSS_BY_LS_PLOTS = PATH_LOSS_PLOTS + "/by_ls_size"
PATH_LOSS_BY_MODEL_PLOTS = PATH_LOSS_PLOTS + "/by_model_name"

# Path Sample outputs
SAMPLE_OUTPUT_ROOT_DIRECTORY = MODELS_PATH + "/sample_outputs"
EVOLUTION_SAMPLE_DIRECTORY = MODELS_PATH + "/evolution_sample"

# Path evolution
EVOLUTION_LENGTH = 100
EVOLUTION_NUMBER_CHANGE = 2

# Path reproduction
REPRODUCED_IMAGES_OUTPUT_DIRECTORY = MODELS_PATH + "/reproduced_images"
IMAGE_TO_REPRODUCE = DATASET_PATH + "/2.png"
REPRODUCED_IMAGE_SUFFIX = "reproduced_image"
