from __future__ import annotations

from config import STR_PATH_PLOTS_ROOT_DIRECTORY, NUMBER_EPOCH_TAKEN_COMPARISON, STR_PATH_PLOTS_HEATMAP_EPOCHS, STR_PATH_PLOTS_HEATMAP_MODEL_SIZE, STR_PATH_PLOTS_HEATMAP_LATENT_SPACE_SIZE, STR_PATH_LOSS_PLOTS, STR_PATH_LOSS_PLOTS_BY_LS, STR_PATH_LOSS_PLOTS_BY_MODEL, STR_PATH_PLOTS_NUMBER_PARAMETERS, \
	PLOT_IMAGE_NAMES, PLOT_NAMES, X_LABEL_NAMES, Y_LABEL_NAMES, STATISTICS_CSV_FILENAME

import matplotlib
import colorsys
import statistics

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ganalyzer.misc import *

def generate_combined_statistics_plots():
	stats_by_model = collect_statistics_by_model()

	ensure_plot_directories()

	colors_list_with_names = get_colors_associated(generate_colors(len(stats_by_model)), [name for name in stats_by_model.keys()])

	save_all_comparisons_models()

	plot_combined_losses(colors_list_with_names, stats_by_model)

	plot_current_number_epoch(stats_by_model, Path(STR_PATH_PLOTS_ROOT_DIRECTORY))

	plot_number_parameters(stats_by_model, Path(STR_PATH_PLOTS_NUMBER_PARAMETERS), DISCRIMINATOR_GLOBAL_NAME)
	plot_number_parameters(stats_by_model, Path(STR_PATH_PLOTS_NUMBER_PARAMETERS), GENERATOR_GLOBAL_NAME)

	plot_median_time_per_epoch(stats_by_model, Path(STR_PATH_PLOTS_ROOT_DIRECTORY))

def save_all_comparisons_models():
	print('\n======> Generate heatmap epoch')
	produce_heatmap_epoch()

	print('\n======> Generate heatmap model size')
	produce_heatmap_model_size()

	print('\n======> Generate heatmap latent space')
	produce_heatmap_latent_space()

def produce_heatmap_model_size():
	for current_latent_dimension_generator in LATENT_DIMENSION_GENERATOR_AVAILABLE:
		comparisons_elements = []  # list every model size for that ls
		current_latent_dimension_generator_str = get_ls_name(current_latent_dimension_generator)

		print("====> now on ", current_latent_dimension_generator_str)

		for current_model in ALL_MODELS:
			total_name = current_model + "-" + current_latent_dimension_generator_str
			available_epochs = get_number_epoch_in_given_setting(total_name)
			epoch_name = get_epoch_name(available_epochs)

			new_elem = (total_name, epoch_name)
			comparisons_elements.append(new_elem)

			print("==> Current model : ", current_model, " nb epochs ", epoch_name, " result ", new_elem)

		save_comparisons_models(
			comparisons_elements,
			Path(STR_PATH_PLOTS_HEATMAP_MODEL_SIZE),
			PLOT_NAMES["latent_space_size_comparison_heatmap"].replace("LATENT_SPACE_SIZE", str(current_latent_dimension_generator)),
			PLOT_IMAGE_NAMES["latent_space_size_comparison_heatmap"].replace("LATENT_SPACE_SIZE", str(current_latent_dimension_generator)) + ".png",
		)

def produce_heatmap_latent_space():
	for current_model in ALL_MODELS:
		comparisons_elements = []  # list every ls for that model size
		print("====> now on ", current_model)
		for current_ls in LATENT_DIMENSION_GENERATOR_AVAILABLE:
			current_latent_dimension_generator_str = get_ls_name(current_ls)
			print("==> Current ls ", current_latent_dimension_generator_str)
			total_name = current_model + "-" + current_latent_dimension_generator_str
			epoch_name = get_epoch_name(get_number_epoch_in_given_setting(total_name))
			new_elem = (total_name, epoch_name)
			comparisons_elements.append(new_elem)

		save_comparisons_models(
			comparisons_elements,
			Path(STR_PATH_PLOTS_HEATMAP_LATENT_SPACE_SIZE),
			PLOT_NAMES["model_size_comparison_heatmap"].replace("MODEL_NAME", current_model),
			PLOT_IMAGE_NAMES["model_size_comparison_heatmap"].replace("MODEL_NAME", current_model) + ".png",
		)

def save_comparisons_models(comparisons_elements, directory, title, output_filename):
	directory.mkdir(parents = True, exist_ok = True)
	size = len(comparisons_elements)

	data = get_values_comparisons(size, comparisons_elements)

	# todo detect if only one element differs
	# row_labels = ["real images"] + [elem[0] + "\n" + elem[1] for elem in comparisons_elements]
	# col_labels = [elem[0] + "\n" + elem[1] for elem in comparisons_elements]
	row_labels = ["real images"] + [elem[0] + elem[1] for elem in comparisons_elements]
	col_labels = [elem[0] + elem[1] for elem in comparisons_elements]

	fig, ax = plt.subplots()
	im = ax.imshow(data, cmap = 'gray', interpolation = 'nearest', vmin = 0, vmax = 1)

	ax.set_xticks(np.arange(len(col_labels)))
	ax.set_yticks(np.arange(len(row_labels)))
	ax.set_xticklabels(col_labels)
	ax.set_yticklabels(row_labels)

	plt.setp(ax.get_xticklabels(), rotation = 90, ha = "right", rotation_mode = "anchor")

	plt.colorbar(im)
	plt.title(title)
	ax.set_xlabel(X_LABEL_NAMES["comparison_heatmap"])
	ax.set_ylabel(Y_LABEL_NAMES["comparison_heatmap"])

	for i in range(len(data)):
		for j in range(len(data[0])):
			data_as_percentage = data[i][j] * 100
			if data_as_percentage < 50:
				color = "white"
			else:
				color = "black"

			ax.text(j, i, str(round(data_as_percentage, 1)) + "%", ha = "center", va = "center", color = color, fontsize = 4)

	plt.subplots_adjust(bottom = 0.6)
	# plt.savefig(Path(STR_PATH_PLOTS_ROOT_DIRECTORY, "heatmap.png"), dpi = 300)
	plt.savefig(Path(directory) / output_filename, dpi = 300)
	plt.close()

def get_values_comparisons(size, comparisons_elements):
	result = [[0.0 for _ in range(size)] for _ in range(size + 1)]
	# nb_comparisons

	for current_generator_index in range(size + 1):
		################################################################################# First, generate images
		if current_generator_index == 0:
			real_images = True
			generated_images = get_real_images_sample()
		else:
			real_images = False
			generated_images = get_fake_images_sample(comparisons_elements[current_generator_index - 1][0], comparisons_elements[current_generator_index - 1][1])

		################################################################################# then, see what discriminators could have discriminated them

		for current_discriminator_index in range(size):
			model_name = comparisons_elements[current_discriminator_index][0]
			epoch = comparisons_elements[current_discriminator_index][1]
			accuracy = get_accuracy_on_images(model_name, epoch, generated_images, real_images)
			result[current_generator_index][int(current_discriminator_index)] = accuracy

	return result

def produce_heatmap_epoch():
	available_settings = [entry.name for entry in Path(STR_PATH_MODELS_ROOT).iterdir() if entry.is_dir()]
	if Path(STR_PATH_PLOTS_ROOT_DIRECTORY).name in available_settings:
		available_settings.remove(Path(STR_PATH_PLOTS_ROOT_DIRECTORY).name)

	for current_setting in available_settings:
		print("====> Current setting : ", current_setting)
		max_epoch = get_number_epoch_in_given_setting(current_setting)
		if max_epoch == 100:
			print("==> Has 100 epoch")
			step = int(max_epoch / NUMBER_EPOCH_TAKEN_COMPARISON)
			current_epoch = 0
			comparisons_elements = []
			for i in range(NUMBER_EPOCH_TAKEN_COMPARISON + 1):
				epoch_name = get_epoch_name(current_epoch)
				comparisons_elements.append((current_setting, epoch_name))
				current_epoch = current_epoch + step

			save_comparisons_models(
				comparisons_elements,
				Path(STR_PATH_PLOTS_HEATMAP_EPOCHS),
				PLOT_NAMES["comparison_heatmap"].replace("MODEL_NAME", current_setting),
				PLOT_IMAGE_NAMES["comparison_heatmap"].replace("MODEL_NAME", current_setting) + ".png",
			)

def collect_statistics_by_model():
	stats_by_model = {}

	if not Path(STR_PATH_MODELS_ROOT).is_dir():
		return stats_by_model

	for directory_path in sorted(Path(STR_PATH_MODELS_ROOT).iterdir()):
		csv_path = directory_path / STATISTICS_CSV_FILENAME
		stats = load_statistics(csv_path)
		stats_by_model[directory_path.name] = stats

	return stats_by_model

def ensure_plot_directories():
	for directory in (
			Path(STR_PATH_PLOTS_ROOT_DIRECTORY),
			Path(STR_PATH_PLOTS_NUMBER_PARAMETERS),
			Path(STR_PATH_PLOTS_HEATMAP_EPOCHS),
			Path(STR_PATH_PLOTS_HEATMAP_MODEL_SIZE),
			Path(STR_PATH_PLOTS_HEATMAP_LATENT_SPACE_SIZE),
			Path(STR_PATH_LOSS_PLOTS),
			Path(STR_PATH_LOSS_PLOTS_BY_LS),
			Path(STR_PATH_LOSS_PLOTS_BY_MODEL),
	):
		directory.mkdir(parents = True, exist_ok = True)

def plot_loss_series(all_colors, series, output_path, title):
	output_path.parent.mkdir(parents = True, exist_ok = True)
	plt.figure(figsize = (12, 6))

	for label, values in series:
		epochs = range(1, len(values) + 1)
		plt.plot(epochs, values, label = label, color = all_colors[label])

	plt.title(title)
	plt.xlabel(X_LABEL_NAMES["loss"])
	plt.ylabel(Y_LABEL_NAMES["loss"])
	if len(series) > 1:
		plt.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(output_path, format = "jpg")
	plt.close()

def plot_combined_losses(color_list, stats_by_model):
	generator_series = []
	discriminator_series = []

	for model_name, stats in stats_by_model.items():  ##TODO refactor the handling of statistics, not good
		if stats.generator_loss:
			generator_series.append((model_name, stats.generator_loss))
		if stats.discriminator_loss:
			discriminator_series.append((model_name, stats.discriminator_loss))

	# original
	plot_loss_series(color_list, generator_series, Path(STR_PATH_LOSS_PLOTS) / f"{PLOT_IMAGE_NAMES['every_generator_loss']}.jpg", PLOT_NAMES['every_generator_loss'])
	plot_loss_series(color_list, discriminator_series, Path(STR_PATH_LOSS_PLOTS) / f"{PLOT_IMAGE_NAMES['every_discriminator_loss']}.jpg", PLOT_NAMES['every_discriminator_loss'])

	# by model_sizes
	plot_split_by_model_size(generator_series, discriminator_series)

	# by ls_size
	plot_split_by_ls_size(generator_series, discriminator_series)

def plot_split_series(series, split_name, output_path, image_name_template, title_template):
	colors = get_colors_associated(generate_colors(len(series)), [name for name, _values in series])
	plot_loss_series(
		colors,
		series,
		output_path / (image_name_template.replace("MODEL_NAME", split_name) + ".jpg"),
		title_template.replace("MODEL_NAME", split_name),
	)

def plot_split_by_generic(generator_series, discriminator_series, split_names, output_path, series_matches_split):
	for current_split_name in split_names:
		print('===> Current plot generator', current_split_name)
		_split_generator_series = [series for series in generator_series if series_matches_split(series[0], current_split_name)]
		plot_split_series(_split_generator_series, current_split_name, output_path, PLOT_IMAGE_NAMES["split_generator_loss"], PLOT_NAMES["split_generator_loss"])

		print('===> Current plot discriminator', current_split_name)
		_split_discriminator_series = [series for series in discriminator_series if series_matches_split(series[0], current_split_name)]
		plot_split_series(_split_discriminator_series, current_split_name, output_path, PLOT_IMAGE_NAMES["split_discriminator_loss"], PLOT_NAMES["split_discriminator_loss"])

def plot_split_by_model_size(generator_series, discriminator_series):
	plot_split_by_generic(
		generator_series,
		discriminator_series,
		ALL_MODELS,
		Path(STR_PATH_LOSS_PLOTS_BY_MODEL),
		lambda series_name, current_plot_model: series_name.startswith(current_plot_model),
	)

def plot_split_by_ls_size(generator_series, discriminator_series):
	ls_sizes_as_string = [get_ls_name(curr_ls) for curr_ls in LATENT_DIMENSION_GENERATOR_AVAILABLE]
	plot_split_by_generic(
		generator_series,
		discriminator_series,
		ls_sizes_as_string,
		Path(STR_PATH_LOSS_PLOTS_BY_LS),
		lambda series_name, current_plot_ls_size: series_name.endswith(current_plot_ls_size),
	)

def get_number_parameters(model_name, model_type):
	model_path = get_model_files_directory(model_name)
	complete_models_list = sorted([path for path in model_path.iterdir() if path.is_file()])

	if model_type == DISCRIMINATOR_GLOBAL_NAME:
		total_path = complete_models_list[0]
	else:
		total_path = complete_models_list[-1]

	model = load_keras_model(total_path)
	nb_params = sum([layer.count_params() for layer in model.layers if layer.trainable])
	return nb_params

def get_contrasting_text_color(background_color):
	red, green, blue, _alpha = background_color
	background_intensity = (red + green + blue) / 3

	return 'black' if background_intensity > 0.5 else 'white'

def produce_heatmap(stats_by_model, output_dir, title, output_filename, value_getter, text_formatter):
	output_dir.mkdir(parents = True, exist_ok = True)
	data = np.zeros((len(ALL_MODELS), len(LATENT_DIMENSION_GENERATOR_AVAILABLE)))

	for model_name, stats in stats_by_model.items():
		idx_x, idx_y = get_model_indexes(model_name)
		data[idx_x, idx_y] = value_getter(model_name, stats)

	x_labels = LATENT_DIMENSION_GENERATOR_AVAILABLE
	y_labels = ALL_MODELS

	plt.figure(figsize = (6, 5))

	heatmap = plt.imshow(data, cmap = 'grey')

	plt.title(title)
	plt.xlabel(X_LABEL_NAMES["heatmap"])
	plt.ylabel(Y_LABEL_NAMES["heatmap"])

	plt.xticks(ticks = np.arange(len(x_labels)), labels = x_labels)
	plt.yticks(ticks = np.arange(len(y_labels)), labels = y_labels)

	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			background_color = heatmap.cmap(heatmap.norm(data[i, j]))
			color = get_contrasting_text_color(background_color)
			plt.text(j, i, text_formatter(data[i, j]), ha = 'center', va = 'center', color = color)

	plt.colorbar(heatmap)

	plt.savefig(output_dir / output_filename, format = 'jpg', dpi = 300)

	plt.show()

def plot_current_number_epoch(stats_by_model, output_dir):
	produce_heatmap(
		stats_by_model,
		output_dir,
		PLOT_NAMES["current_number_epoch"],
		PLOT_IMAGE_NAMES["current_number_epoch"] + ".jpg",
		lambda _model_name, stats: len(stats.epoch_durations),
		text_formatter = lambda value: str(int(value)),
	)

def plot_number_parameters(stats_by_model, output_dir, model_type):
	produce_heatmap(
		stats_by_model,
		output_dir,
		PLOT_NAMES["number_parameters"].replace("MODEL_NAME", model_type),
		PLOT_IMAGE_NAMES["number_parameters"].replace("MODEL_NAME", model_type) + ".jpg",
		lambda model_name, _stats: int(get_number_parameters(model_name, model_type)),
		text_formatter = lambda value: f"{int(value):,d}".replace(",", " "),
	)

def plot_median_time_per_epoch(stats_by_model, output_dir):  # todo merge with time taken
	produce_heatmap(
		stats_by_model,
		output_dir,
		PLOT_NAMES["median_time_per_epoch"],
		PLOT_IMAGE_NAMES["median_time_per_epoch"] + ".jpg",
		lambda _model_name, stats: statistics.median(stats.epoch_durations),
		text_formatter = lambda value: str(round(value, 2)),
	)

def get_colors_associated(colors_list, stats):
	result = {}
	current_index = 0

	for name in stats:
		result[name] = colors_list[current_index]
		current_index += 1

	return result

def generate_colors(n):
	colors = []
	for i in range(n):
		r, g, b = colorsys.hsv_to_rgb(i / n, 1.0, 1.0)

		colors.append("#{0:02x}{1:02x}{2:02x}".format(int(r * 255), int(g * 255), int(b * 255)))

	return colors
