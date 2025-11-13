from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf

logger = logging.getLogger(__name__)

class ModelViewer:

	def __init__(self, models_list, previous_height, parent, name, n_col, is_output_image, calling_context):
		self.parent = parent
		self.name = name
		self.n_col = n_col
		self.previous_panels_height = previous_height
		self.is_output_image = is_output_image
		self.calling_context = calling_context

		self.image_size = 150
		self.debounce_time = 50
		self.debounce_id = None

		self.models_list = models_list
		self.models_quantity = len(self.models_list)
		if self.models_quantity == 0:
			raise ValueError("ModelViewer requires at least one model to display.")

		self.current_model = None
		self.current_input = np.full(1, 1)

		# Initialize UI
		self.initialize_title()
		self.initialize_layout()
		self.initialize_epoch_layout()
		self.image_input_data = self.create_labeled_image(1, "Input")
		self.initialize_inside_viewer()
		self.image_output_data = self.create_labeled_image(3, "Output")

		self.slider_epoch.set(self.models_quantity - 1)
		self.on_epoch_slider_change(self.models_quantity - 1)
		self.update_inside_selector()

	def initialize_title(self):
		title_generator_hint = tk.Label(self.parent, text = self.name, bg = "#666666")
		title_generator_hint.grid(row = self.previous_panels_height, column = 0, rowspan = 1, columnspan = self.n_col, sticky = "we")

	def initialize_layout(self):
		self.layout_panel = tk.Frame(self.parent, bg = "#444444")
		self.layout_panel.grid(row = self.previous_panels_height + 1, column = 0, rowspan = 1, columnspan = self.n_col, sticky = "nsew")
		self.layout_panel.columnconfigure((0, 1, 2, 3), weight = 1)
		self.layout_panel.columnconfigure(2, weight = 5)
		self.layout_panel.rowconfigure(0, weight = 1)

	def initialize_epoch_layout(self):
		layout_epoch = tk.Frame(self.layout_panel, bg = "#111111")
		layout_epoch.grid(row = 0, column = 0, rowspan = 1, columnspan = 1, sticky = "nsew")
		layout_epoch.columnconfigure(0, weight = 1)
		layout_epoch.rowconfigure(0, weight = 1)
		layout_epoch.rowconfigure(1, weight = 1)

		self.label_current_epoch = tk.Label(layout_epoch)
		self.label_current_epoch.grid(row = 0, column = 0, columnspan = 1, sticky = "nsew")

		self.slider_epoch = ttk.Scale(layout_epoch, from_ = 0, to = self.models_quantity - 1, orient = "horizontal", command = self.on_epoch_slider_change_debounced)
		self.slider_epoch.grid(row = 1, column = 0, columnspan = 1)

	def get_layers_list(self):
		if not self.current_model:
			return []

		return [f"{i}) {layer.name}" for i, layer in enumerate(self.current_model.layers)]

	def get_current_layer_index(self):
		if not getattr(self, "selected_inside_layer", ""):
			raise ValueError("No layer selected")

		return int(self.selected_inside_layer.split(")", maxsplit = 1)[0])

	def create_labeled_image(self, position, name):
		temp_panel = tk.Frame(self.layout_panel, bg = "#222222")
		temp_panel.grid(column = position, row = 0, rowspan = 1, columnspan = 1, sticky = "nsew")
		temp_panel.columnconfigure(0, weight = 1)
		temp_panel.rowconfigure(0, weight = 1)
		temp_panel.rowconfigure(1, weight = 5)

		image_label = tk.Label(temp_panel, text = name)
		image_label.grid(row = 0, column = 0, rowspan = 1, columnspan = 1, sticky = "nsew")

		image_model = tk.Label(temp_panel)
		image_model.grid(row = 1, column = 0, rowspan = 1, columnspan = 1, sticky = "nsew")

		return image_model

	def initialize_inside_viewer(self):

		inside_viewer_layout = tk.Frame(self.layout_panel, bg = "#ff0000")
		inside_viewer_layout.grid(row = 0, column = 2, rowspan = 1, columnspan = 1, sticky = "nsew")
		inside_viewer_layout.columnconfigure(0, weight = 1)
		inside_viewer_layout.rowconfigure(0, weight = 1)
		inside_viewer_layout.rowconfigure(1, weight = 6)

		self.inside_selector = ttk.Combobox(inside_viewer_layout, state = "readonly")
		self.inside_selector.set("Select Location")
		self.inside_selector.grid(row = 0, column = 0, columnspan = 1, sticky = "nsew")
		self.inside_selector.bind("<<ComboboxSelected>>", self.on_selector_layer_change)

		self.image_inside_data = tk.Label(inside_viewer_layout, bg = "#0000ff")
		self.image_inside_data.grid(row = 1, column = 0, rowspan = 1, columnspan = 1, sticky = "nsew")

	def update_inside_selector(self):

		layers_list = self.get_layers_list()
		self.inside_selector.config(values = layers_list)
		if layers_list:
			self.selected_inside_layer = layers_list[0]
			self.inside_selector.set(layers_list[0])
		else:
			self.selected_inside_layer = ""
			self.inside_selector.set("Select Location")

	def on_selector_layer_change(self, event):  # pragma: no cover - tkinter callback
		self.selected_inside_layer = event.widget.get()
		self.refresh_inside_visualization()

	def refresh_inside_visualization(self):

		if self.current_input.ndim <= 1:
			logger.warning("%s input not available for visualization", self.name)
			return

		self.refresh_layer_visualization()

	def refresh_layer_visualization(self):
		if not self.current_model:
			logger.warning("Cannot refresh layer visualization without a model.")
			return

		try:
			index_layer = self.get_current_layer_index()
			layer = self.current_model.layers[index_layer]
		except (ValueError, IndexError):
			logger.warning("Invalid layer selected for visualization.")
			return

		try:
			intermediate = tf.keras.Model(inputs = self.current_model.inputs, outputs = layer.output)
			layer_output = intermediate.predict(self.current_input)
		except Exception:  # pragma: no cover - defensive log
			logger.exception("Failed to compute intermediate output for layer %s", layer.name)
			return

		if layer_output.ndim == 4:
			representation = self.get_array_representation(layer_output[0])
			is_color = representation.ndim == 3 and representation.shape[-1] in {3, 4}
		elif layer_output.ndim == 2:
			representation = self.get_rectangle_representation(layer_output[0])
			is_color = False
		else:
			logger.warning("Unsupported output shape %s for visualization", layer_output.shape)
			return

		self.refresh_tk_image(representation, is_color = is_color, tk_image = self.image_inside_data)

	def get_array_representation(self, raw_data):
		if raw_data.ndim == 3 and raw_data.shape[-1] > 1:
			raw_data = raw_data.mean(axis = -1)

		min_val = np.min(raw_data)
		max_val = np.max(raw_data)
		if max_val > min_val:
			normalized = (raw_data - min_val) / (max_val - min_val)
		else:
			normalized = np.zeros_like(raw_data)

		return (normalized * 255).astype(np.uint8)

	def get_rectangle_representation(self, raw_data):

		flat = raw_data.flatten()
		size = int(np.ceil(np.sqrt(flat.size)))
		padded = np.zeros(size * size, dtype = flat.dtype)
		padded[: flat.size] = flat
		grid = padded.reshape((size, size))

		min_val = np.min(grid)
		max_val = np.max(grid)
		if max_val > min_val:
			normalized = (grid - min_val) / (max_val - min_val)
		else:
			normalized = np.zeros_like(grid)

		return (normalized * 255).astype(np.uint8)

	def refresh_tk_image(self, input_matrix, is_color, tk_image):

		if input_matrix.size == 0:
			logger.warning("No data available to display.")
			return

		if is_color:
			img = Image.fromarray(input_matrix.astype("uint8"), mode = "RGB")
		else:
			img = Image.fromarray(input_matrix.astype("uint8"), mode = "L")

		img_tk = ImageTk.PhotoImage(img.resize((self.image_size, self.image_size), Image.NEAREST))
		tk_image.configure(image = img_tk)
		tk_image.image = img_tk

	def refresh_data_in(self, value):  # pragma: no cover - awaiting implementation
		pass

	def refresh_data_out(self, value):  # pragma: no cover - awaiting implementation
		if self.is_output_image:
			pass
		else:
			pass

	def on_epoch_slider_change_debounced(self, value):  # pragma: no cover - tkinter callback
		if self.debounce_id:
			self.parent.after_cancel(self.debounce_id)

		# Schedule the real handler to run after debounce_time
		self.debounce_id = self.parent.after(self.debounce_time, self.on_epoch_slider_change, value)

	def on_epoch_slider_change(self, value):

		new_epoch_exact = int(float(value))

		try:
			new_epoch_found = self.get_closest_model_loaded_index(new_epoch_exact)
		except ValueError:
			logger.error("Unable to find a loaded model near index %s", new_epoch_exact)
			return

		self.current_model = self.models_list[new_epoch_found]
		self.update_inside_selector()

		if self.name == "Discriminator":
			self.calling_context.update_discriminator()
		elif self.name == "Generator":
			self.calling_context.update_generator()
		else:
			logger.warning("Unknown model viewer name '%s'", self.name)

		self.label_current_epoch.config(text = f"Current Epoch : {new_epoch_found} / {self.models_quantity - 1}")

	def get_closest_model_loaded_index(self, model_index):

		if 0 <= model_index < self.models_quantity and self.models_list[model_index]:
			return model_index

		lower = model_index - 1
		upper = model_index + 1
		while lower >= 0 or upper < self.models_quantity:
			if lower >= 0 and self.models_list[lower]:
				return lower
			if upper < self.models_quantity and self.models_list[upper]:
				return upper
			lower -= 1
			upper += 1

		raise ValueError("No models available in the provided list.")
