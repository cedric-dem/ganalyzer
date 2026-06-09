from ganalyzer.misc import convert_keras_to_tflite
from ganalyzer.utils_sample import produce_continuous_movement, produce_evolution_sample, produce_sample_outputs
from ganalyzer.utils_plot import generate_combined_statistics_plots

if __name__ == "__main__":
	convert_keras_to_tflite()

	produce_continuous_movement()
	produce_evolution_sample()
	produce_sample_outputs()

	generate_combined_statistics_plots()
