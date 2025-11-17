from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Iterable, List
import copy

from ganalyzer.ModelViewer import ModelViewer
from ganalyzer.misc import project_array
from config import latent_dimension_generator, model_name, rgb_images
import numpy as np

class GUI:
	def __init__(self, models_list_generator, models_list_discriminator):
		self.n_col = 15
		self.n_row = 12
		self.image_size = 150

		self.default_value_k = 0
		self.default_value_mu = 0
		self.default_value_sigma = 1

		self.models_quantity = min(len(models_list_generator), len(models_list_discriminator))

		self.root = tk.Tk()
		self.root.configure(bg = "black")
		self.root.title("GANalyzer")

		self.initializing = True

		self._initialize_input_panel()

		self.generator_viewer = ModelViewer(models_list_generator, 2, self.root, "Generator", 15, True, self)

		self.discriminator_viewer = ModelViewer(models_list_discriminator, 17, self.root, "Discriminator", 15, False, self)

		self._init_selectors()

		self.initializing = False

		self.update_generator()

		self.root.mainloop()

	def _init_selectors(self):
		self._refresh_label_k(self.default_value_k)
		self._refresh_label_mu(self.default_value_mu)
		self._refresh_label_sigma(self.default_value_sigma)

		self.randomize_all_sliders(self.default_value_mu, self.default_value_sigma)

	def _initialize_input_panel(self):
		title_input_hint = tk.Label(self.root, text = "Input", bg = "#666666")
		title_input_hint.grid(row = 0, column = 0, columnspan = self.n_col, sticky = "we")

		self.input_image_grid_size = int(latent_dimension_generator ** 0.5)
		self.max_slider_value = 5

		notebook = ttk.Notebook(self.root)
		notebook.grid(row = 1, column = 0, columnspan = self.n_col, sticky = "nsew")

		random_input_tab = self._create_tab(notebook, "Set Random Inputs", (0, 1, 2, 3))
		constant_input_tab = self._create_tab(notebook, "Set Constant Input", (0, 1, 2, 3))
		manual_input_tab = self._create_manual_input_tab(notebook)

		self._build_manual_input_controls(manual_input_tab)
		self._build_constant_controls(constant_input_tab)
		self._build_random_controls(random_input_tab)

	def _create_tab(self, notebook, title, rows):
		tab = ttk.Frame(notebook)
		notebook.add(tab, text = title)
		tab.columnconfigure((0, 1, 2, 3, 4, 5), weight = 1)
		tab.rowconfigure(rows, weight = 1)
		return tab

	def _create_manual_input_tab(self, notebook):
		tab = ttk.Frame(notebook)
		notebook.add(tab, text = "Set Input Manually")
		indices = tuple(range(self.input_image_grid_size))
		tab.columnconfigure(indices, weight = 1)
		tab.rowconfigure(indices, weight = 1)
		return tab

	def _build_manual_input_controls(self, parent):
		self.slider_grid: List[List[ttk.Scale]] = [
			[self._create_grid_slider(i, j, parent) for j in range(self.input_image_grid_size)]
			for i in range(self.input_image_grid_size)
		]

	def _build_constant_controls(self, parent):
		label_hint_constant = tk.Label(parent, text = "Set All Constant Value : ")
		label_hint_constant.grid(row = 0, column = 1, columnspan = 2, sticky = "nsew")

		self.label_k = self._create_parameter_input_label(1, 1, parent)
		self.slider_k = self._create_parameter_input_slider(self.default_value_k, 3, 1, True, self._refresh_label_k, parent)

		button_set_input_constant = ttk.Button(parent, text = "Set", command = self.set_input_constant)
		button_set_input_constant.grid(row = 3, column = 1, columnspan = 2)

	def _build_random_controls(self, parent):
		label_hint_random = tk.Label(parent, text = "Set All Random Value ")
		label_hint_random.grid(row = 0, column = 1, columnspan = 2, sticky = "nsew")

		self.label_mu = self._create_parameter_input_label(1, 1, parent)
		self.label_sigma = self._create_parameter_input_label(1, 2, parent)

		self.slider_mu = self._create_parameter_input_slider(self.default_value_mu, 3, 1, True, self._refresh_label_mu, parent)
		self.slider_sigma = self._create_parameter_input_slider(self.default_value_sigma, 3, 2, False, self._refresh_label_sigma, parent)

		button_set_input_random = ttk.Button(parent, text = "Set", command = self.set_input_random)
		button_set_input_random.grid(row = 3, column = 1, columnspan = 2)

	def _create_grid_slider(self, i, j, parent):
		slider = ttk.Scale(parent, from_ = -self.max_slider_value, to = self.max_slider_value, orient = "horizontal", length = 100)
		slider.grid(row = i + 1, column = j, padx = 3, pady = 3)
		slider.bind("<ButtonRelease-1>", self.update_generator)
		return slider

	def generate_image_from_input_values(self, input_raw):
		self.generator_viewer.current_input = np.array([input_raw])

		if rgb_images:
			predicted_raw = self.generator_viewer.current_model(self.generator_viewer.current_input, training = False)[0, :, :, :]
		else:
			predicted_raw = self.generator_viewer.current_model(self.generator_viewer.current_input)[0, :, :, 0]

		return np.round(project_array(predicted_raw, 254, -1, 1)).astype(np.uint8)

	def update_generator(self, _event = None):
		if self.initializing:
			return

		values = self._get_manual_input_values()

		input_before_reshape = np.array(values).reshape((self.input_image_grid_size, self.input_image_grid_size))
		input_after_reshape = project_array(input_before_reshape, 254, -self.max_slider_value, self.max_slider_value).astype(np.uint8)
		self.generator_viewer.refresh_tk_image(input_after_reshape, False, self.generator_viewer.image_input_data)

		self.generator_viewer.refresh_inside_visualization()

		self.generated_image = self.generate_image_from_input_values(values)
		self.generator_viewer.refresh_tk_image(self.generated_image, rgb_images, self.generator_viewer.image_output_data)

		self.update_discriminator()

	def _get_manual_input_values(self):
		return [slider.get() for row in self.slider_grid for slider in row]

	def update_discriminator(self):
		if self.initializing:
			return

		self.discriminator_viewer.refresh_tk_image(self.generated_image, rgb_images, self.discriminator_viewer.image_input_data)

		self.discriminator_viewer.refresh_inside_visualization()
		self._refresh_prediction_discriminator()

	def _refresh_prediction_discriminator(self):
		self.discriminator_viewer.current_input = np.array([((self.generated_image - 127.5) / 127.5).astype(np.float64)])
		predicted_output = 0
		if model_name in {"test_0", "test_0B"}:
			predicted_output = self.discriminator_viewer.current_model.predict(self.discriminator_viewer.current_input)[0][0]
		elif model_name == "test_1":
			predicted_output = self.discriminator_viewer.current_model.predict(self.discriminator_viewer.current_input)[0][0][0][0]
		self.discriminator_viewer.image_output_data.config(text = "Prediction : " + str(round(predicted_output, 2)))

	def randomize_all_sliders(self, mu, sigma):
		for row in self.slider_grid:
			for slider in row:
				random_value = np.random.normal(loc = mu, scale = sigma)
				val_clipped = max(-self.max_slider_value, min(self.max_slider_value, random_value))
				slider.set(val_clipped)

		self.update_generator()

	def set_input_constant(self):
		new_k_value = self.slider_k.get()
		for row in self.slider_grid:
			for slider in row:
				slider.set(new_k_value)

		self.update_generator()

	def set_input_random(self):
		new_mu_value = self.slider_mu.get()
		new_sigma_value = self.slider_sigma.get()
		self.randomize_all_sliders(new_mu_value, new_sigma_value)

	def _create_parameter_input_label(self, x, y, parent):
		label = tk.Label(parent)
		label.grid(row = y, column = x, columnspan = 2)
		return label

	def _create_parameter_input_slider(self, default_value, x, y, can_be_negative, method_refresh_text, parent):
		min_value = -self.max_slider_value if can_be_negative else 0

		slider = ttk.Scale(parent, from_ = min_value, to = self.max_slider_value, orient = "horizontal", length = 100, command = method_refresh_text)

		slider.grid(row = y, column = x)
		slider.set(default_value)

		return slider

	def _refresh_label_k(self, value):
		self.label_k.config(text = "K = " + str(round(float(value), 2)))

	def _refresh_label_mu(self, value):
		self.label_mu.config(text = "Mu = " + str(round(float(value), 2)))

	def _refresh_label_sigma(self, value):
		self.label_sigma.config(text = "Sigma = " + str(round(float(value), 2)))
