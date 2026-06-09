MODEL_OUTPUT_SIZE = 100

ALL_MODELS = ["model_0_small", "model_1_medium", "model_2_large"]
MODEL_NAME = ALL_MODELS[0]

LATENT_DIMENSION_GENERATOR_AVAILABLE = [49, 100, 196]
LATENT_DIMENSION_GENERATOR = LATENT_DIMENSION_GENERATOR_AVAILABLE[0]

IS_RGB_IMAGES = True

IMAGE_NORMALIZATION_CENTER = 127.5
IMAGE_NORMALIZATION_SCALE = 127.5

# train
BATCH_SIZE = 32
SAVE_TRAIN_EPOCH_EVERY = 10

# inverse generator train
INVERSE_GENERATOR_TRAIN_EPOCHS = 100
STEPS_PER_INVERSE_GENERATOR_EPOCH = 200
AUTOENCODER_PIXEL_DIFFERENCE_BAR_COUNT = 100
INVERSE_GENERATOR_MODEL_TYPE = "inverse_generator"
INVERSE_GENERATOR_DIRECTORY_NAME = "inverse_generator"
INVERSE_COMPARISON_DIRECTORY_NAME = "comparison"
INVERSE_PLOTS_DIRECTORY_NAME = "plots_inverse"
INVERSE_GENERATOR_STATISTICS_HEADERS = ("epoch_id", "time", "loss_mse", "loss_mae")

# inverse generator plot names
INVERSE_GENERATOR_PLOT_NAMES = {
	"autoencoder_pixel_difference": "Autoencoder absolute pixel difference",
	"generator_inverse_latent_difference": "Generator/inverse generator absolute latent-vector difference",
	"inverse_loss_mse": "Inverse Generator MSE Loss Over Epochs",
	"inverse_loss_mae": "Inverse Generator MAE Loss Over Epochs",
}

INVERSE_GENERATOR_PLOT_IMAGE_NAMES = {
	"autoencoder_pixel_difference": "autoencoder_pixel_difference.png",
	"generator_inverse_latent_difference": "generator_inverse_latent_difference.png",
	"inverse_loss_mse": "loss_mse.png",
	"inverse_loss_mae": "loss_mae.png",
}

INVERSE_GENERATOR_X_LABEL_NAMES = {
	"autoencoder_pixel_difference": "Absolute pixel difference (%)",
	"generator_inverse_latent_difference": "Absolute latent-vector difference",
	"inverse_loss_mse": "Epoch",
	"inverse_loss_mae": "Epoch",
}

INVERSE_GENERATOR_Y_LABEL_NAMES = {
	"autoencoder_pixel_difference": "Pixels (%)",
	"generator_inverse_latent_difference": "Latent vector values (%)",
	"inverse_loss_mse": "Loss MSE",
	"inverse_loss_mae": "Loss MAE",
}

AUTOENCODER_PIXEL_DIFFERENCE_BARPLOT_FILENAME = INVERSE_GENERATOR_PLOT_IMAGE_NAMES["autoencoder_pixel_difference"]
INVERSE_LATENT_DIFFERENCE_BARPLOT_FILENAME = INVERSE_GENERATOR_PLOT_IMAGE_NAMES["generator_inverse_latent_difference"]
INVERSE_LOSS_MSE_PLOT_FILENAME = INVERSE_GENERATOR_PLOT_IMAGE_NAMES["inverse_loss_mse"]
INVERSE_LOSS_MAE_PLOT_FILENAME = INVERSE_GENERATOR_PLOT_IMAGE_NAMES["inverse_loss_mae"]

# GUI
LOAD_QUANTITY_GUI = 3  # max is around 160 for medium on my rtx 3060

# statistics
NUMBER_EPOCH_TAKEN_COMPARISON = 5
NUMBER_COMPARISON = 100

# NAMING
DISCRIMINATOR_GLOBAL_NAME = "discriminator"
GENERATOR_GLOBAL_NAME = "generator"
EPOCH_GLOBAL_NAME = "epoch"
LATENT_SPACE_GLOBAL_NAME = "ls"

# sample
SAMPLE_QUANTITY = 20

# prefix
CONTINUOUS_MOVEMENT_IMAGE_PREFIX = "continuous_movement"
EVOLUTION_SAMPLE_PREFIX = "evolution_sample"
SAMPLE_OUTPUT_PREFIX = "sample_output_epoch_"

# continuous movement
CONTINUOUS_MOVEMENT_LENGTH = 100
CONTINUOUS_MOVEMENT_NUMBER_CHANGES = 2

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
STR_PATH_DATASET = "datasets/" + DATASET_NAME + "/" + DATASET_DIMENSION

RESULTS_ROOT_PATH = "results/" + DATASET_NAME
STR_PATH_MODELS_ROOT = RESULTS_ROOT_PATH + "/models"
MODELS_PATH = STR_PATH_MODELS_ROOT + "/" + MODEL_NAME_EXPLICIT
MODELS_DIRECTORY_NAME = "models_files"
MODELS_DIRECTORY = MODELS_PATH + "/" + MODELS_DIRECTORY_NAME

MODELS_DIRECTORY_NAME_INVERSE = "models_files_inverse"

MODELS_AS_TFLITE = MODELS_PATH + "/models_as_tf_lite"

STATISTICS_CSV_FILENAME = "statistics.csv"
STATISTICS_FILE_PATH = MODELS_PATH + "/" + STATISTICS_CSV_FILENAME

# Path plot
STR_PATH_PLOTS_ROOT_DIRECTORY = RESULTS_ROOT_PATH + "/plots"

STR_PATH_PLOTS_HEATMAP_EPOCHS = STR_PATH_PLOTS_ROOT_DIRECTORY + "/heatmap_epochs"
STR_PATH_PLOTS_HEATMAP_MODEL_SIZE = STR_PATH_PLOTS_ROOT_DIRECTORY + "/heatmap_model_size"
STR_PATH_PLOTS_HEATMAP_LATENT_SPACE_SIZE = STR_PATH_PLOTS_ROOT_DIRECTORY + "/heatmap_latent_space_size"
STR_PATH_PLOTS_NUMBER_PARAMETERS = STR_PATH_PLOTS_ROOT_DIRECTORY + "/number_parameters"

STR_PATH_LOSS_PLOTS = STR_PATH_PLOTS_ROOT_DIRECTORY + "/loss"

STR_PATH_LOSS_PLOTS_BY_LS = STR_PATH_LOSS_PLOTS + "/by_ls_size"
STR_PATH_LOSS_PLOTS_BY_MODEL = STR_PATH_LOSS_PLOTS + "/by_model_name"

# Path Sample outputs
SAMPLE_OUTPUT_ROOT_DIRECTORY = MODELS_PATH + "/sample_outputs"

# Path continuous movement
CONTINUOUS_MOVEMENT_DIRECTORY = MODELS_PATH + "/continuous_movement"

# Path evolution sample
EVOLUTION_SAMPLE_PATH = MODELS_PATH + "/evolution_sample"
