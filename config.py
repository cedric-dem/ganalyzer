model_name = "test_0B"

model_path = "model_" + model_name + "/"

statistics_file_path = model_path + "statistics.csv"

rgb_images = True

latent_dimension_generator = 169

show_models_gui=10

use_full_dataset = True

if use_full_dataset:
    dataset_path = "dataFull/"
else:
    dataset_path = "dataReduced/"
