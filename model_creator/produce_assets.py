from config import LATENT_DIMENSION_GENERATOR_AVAILABLE, ALL_MODELS
from ganalyzer.misc import convert_keras_to_tflite
from ganalyzer.utils_sample import produce_continuous_movement, produce_evolution_sample, produce_sample_outputs
from ganalyzer.utils_plot import generate_combined_statistics_plots

if __name__ == "__main__":

	print("=> Producing split assets")
	for curr_model_name in ALL_MODELS:
		for curr_latent_space_size in LATENT_DIMENSION_GENERATOR_AVAILABLE:
			print(f"==> generating sample outputs for model {curr_model_name} with latent space size {curr_latent_space_size}")

			convert_keras_to_tflite(curr_model_name, curr_latent_space_size)

			produce_continuous_movement(curr_model_name, curr_latent_space_size)
			produce_evolution_sample(curr_model_name, curr_latent_space_size)
			produce_sample_outputs(curr_model_name, curr_latent_space_size)

	generate_combined_statistics_plots()
