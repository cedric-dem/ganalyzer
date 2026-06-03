from ganalyzer.misc import convert_keras_to_tflite, produce_continuous_movement, produce_evolution_sample, produce_sample_outputs, reproduction_search, _generate_combined_statistics_plots

if __name__ == "__main__":
	convert_keras_to_tflite()
	produce_continuous_movement()
	produce_evolution_sample()
	produce_sample_outputs()
	reproduction_search()
	_generate_combined_statistics_plots()
