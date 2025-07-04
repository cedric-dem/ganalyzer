from misc import *

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf


class ModelViewer(object):
    def __init__(self, models_list, previous_height, parent, name, n_col, is_output_image, calling_context):
        self.parent = parent

        self.image_size = 150
        self.debounce_time = 50
        self.debounce_id = None

        self.current_input = np.full(1, 1)
        self.models_list = models_list
        self.current_model = None
        self.n_col = n_col
        self.models_quantity = len(self.models_list)

        self.previous_panels_height = previous_height

        self.name = name

        self.is_output_image = is_output_image

        # Initialize UI
        self.initialize_title()
        self.initialize_layout()
        self.initialize_epoch_layout()
        self.image_input_data = self.get_image_labeled(1, "Input")
        self.initialize_inside_viewer()
        self.image_output_data = self.get_image_labeled(3, "Output")
        self.calling_context = calling_context

        self.slider_epoch.set(self.models_quantity - 1)
        self.on_epoch_slider_change(self.models_quantity - 1)

        layers_list = self.get_layers_list()
        self.selected_inside_layer = layers_list[0]
        self.inside_selector.config(values=layers_list)

    def initialize_title(self):
        title_generator_hint = tk.Label(self.parent, text=self.name, bg="#666666")
        title_generator_hint.grid(row=self.previous_panels_height, column=0, rowspan=1, columnspan=self.n_col, sticky="we")

    def initialize_layout(self):
        self.layout_panel = tk.Frame(self.parent, bg="#444444")
        self.layout_panel.grid(rowspan=1, columnspan=self.n_col, sticky="nsew")
        self.layout_panel.columnconfigure((0, 1, 2, 3), weight=1)
        self.layout_panel.columnconfigure(2, weight=5)
        self.layout_panel.rowconfigure((0), weight=1)

    def initialize_epoch_layout(self):

        layout_epoch = tk.Frame(self.layout_panel, bg="#111111")
        layout_epoch.grid(rowspan=1, columnspan=1, sticky="nsew")
        layout_epoch.columnconfigure((0), weight=1)
        layout_epoch.rowconfigure((0, 1), weight=1)

        self.label_current_epoch = tk.Label(layout_epoch)
        self.label_current_epoch.grid(row=0, column=0, columnspan=1, sticky="nsew")

        self.slider_epoch = ttk.Scale(layout_epoch, from_=0, to=self.models_quantity - 1, orient="horizontal", command=self.on_epoch_slider_change_debounced)
        self.slider_epoch.grid(row=1, column=0, columnspan=1)

    def get_layers_list(self):
        return [str(i) + ") " + self.current_model.layers[i].name for i in range(len(self.current_model.layers))]

    def get_current_layer_index(self):
        return int(self.selected_inside_layer.split(")")[0])

    def get_image_labeled(self, position, name):

        temp_panel = tk.Frame(self.layout_panel, bg="#222222")
        temp_panel.grid(column=position, row=0, rowspan=1, columnspan=1, sticky="nsew")
        temp_panel.columnconfigure((0,), weight=1)
        temp_panel.rowconfigure(0, weight=1)
        temp_panel.rowconfigure(1, weight=5)

        image_label = tk.Label(temp_panel, text=name)
        image_label.grid(row=0, column=position, rowspan=1, columnspan=1, sticky="nsew")

        image_model = tk.Label(temp_panel)
        image_model.grid(row=1, column=position, rowspan=1, columnspan=1, sticky="nsew")

        return image_model

    def initialize_inside_viewer(self):
        inside_viewer_layout = tk.Frame(self.layout_panel, bg="#ff0000")
        inside_viewer_layout.grid(row=0, column=2, rowspan=1, columnspan=1, sticky="nsew")
        inside_viewer_layout.columnconfigure((0), weight=1)
        inside_viewer_layout.rowconfigure((0, 1), weight=1)
        inside_viewer_layout.rowconfigure(1, weight=6)

        self.inside_selector = ttk.Combobox(inside_viewer_layout, state="readonly")
        self.inside_selector.set("Select Location")
        self.inside_selector.grid(row=0, column=0, columnspan=1, sticky="nsew")
        self.inside_selector.bind("<<ComboboxSelected>>", self.on_selector_layer_change)

        self.image_inside_data = tk.Label(inside_viewer_layout, bg="#0000ff")
        self.image_inside_data.grid(rowspan=1, columnspan=1, sticky="nsew")

    def on_selector_layer_change(self, event):
        self.selected_inside_layer = event.widget.get()

        self.refresh_inside_visualization()

    def refresh_inside_visualization(self):

        print("==> now refreshing ", self.name, " layer ", self.selected_inside_layer)
        if self.current_input.ndim > 1:
            self.refresh_layer_visualization()
        else:
            print("==> Discriminator Input not found")

    def refresh_layer_visualization(self):
        print("==> refresh, having settings")

        index_layer = self.get_current_layer_index()
        print("=======>", self.current_input.shape, self.image_inside_data, self.current_model, index_layer)

        layer_output = tf.keras.Model(inputs=self.current_model.inputs, outputs=self.current_model.layers[index_layer].output).predict(self.current_input)
        print("===> layer ", index_layer, " name : ", self.current_model.layers[index_layer].name, " shape ", layer_output.shape, " min value", np.min(layer_output), " max ", np.max(layer_output))

        if layer_output.ndim == 4:
            representation = self.get_array_representation(layer_output[0])

        elif layer_output.ndim == 2:
            representation = self.get_rectangle_representation(layer_output[0])

        else:
            representation = []
            print("==> Unknown dimension", layer_output.shape)

        self.refresh_tk_image(representation, False, self.image_inside_data)

    def get_array_representation(self, raw_data):
        print("==> Draw lots of squares")
        result = np.full((100, 100), 254, dtype=np.uint8)

        # TODO
        print("==> Case A with shape ", raw_data.shape)


        return result

    def get_rectangle_representation(self, raw_data):
        print("=> Draw one rectangle")
        size=int(raw_data.shape[0]**0.5)
        print("==> Case B with shape ", raw_data.shape, " so ",size)

        raw_data_square=raw_data.reshape((size, size))
        result_r = find_limits_and_project(raw_data_square)

        result = np.full((size+2, size+2), 254, dtype=np.uint8)
        result[1:size + 1, 1:size + 1] = result_r
        return result

    def refresh_tk_image(self, input_matrix, is_color, tk_image: tk.Label):
        if is_color:
            img = Image.fromarray(input_matrix.astype("uint8"), mode="RGB")
        else:
            img = Image.fromarray(input_matrix, mode="L")

        img_tk = ImageTk.PhotoImage(img.resize((self.image_size, self.image_size), Image.NEAREST))
        tk_image.configure(image=img_tk)
        tk_image.image = img_tk

    def refresh_data_in(self, value):
        pass

    def refresh_data_out(self, value):
        if self.is_output_image:
            pass
        else:
            pass

    def on_epoch_slider_change_debounced(self, value):
        if self.debounce_id:
            self.parent.after_cancel(self.debounce_id)

        # Schedule the real handler to run after debounce_time
        self.debounce_id = self.parent.after(self.debounce_time, self.on_epoch_slider_change, value)

    def on_epoch_slider_change(self, value):

        new_epoch_exact = int(float(value))

        new_epoch_found = self.get_closest_model_loaded_index(new_epoch_exact)

        self.current_model = self.models_list[new_epoch_found]

        if self.name == "Discriminator":
            self.calling_context.update_discriminator()
        elif self.name == "Generator":
            self.calling_context.update_generator()
        else:
            print("==> Generator or Discriminator not found")

        self.label_current_epoch.config(text="Current Epoch : " + str(new_epoch_found) + " / " + str(self.models_quantity - 1))

    def get_closest_model_loaded_index(self, model_index):
        if self.models_list[model_index]:
            found_index = model_index
        else:
            current_delta = 0
            found_index = 0
            while found_index == 0:
                for direction in [-1, 1]:
                    new_index = model_index + direction * current_delta
                    if new_index < len(self.models_list) and self.models_list[new_index]:
                        found_index = new_index

                current_delta += 1
        return found_index
