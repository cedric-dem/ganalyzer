model_name = "test_0"

model_path = "model_" + model_name + "/"

statistics_file_path = model_path + "statistics.csv"

rgb_images = True

latent_dimension_generator = 169

if model_name == "test_0B":
    show_models_gui = 50
else:
    show_models_gui = 160

use_full_dataset = True

if use_full_dataset:
    dataset_path = "dataFull/"
else:
    dataset_path = "dataReduced/"
