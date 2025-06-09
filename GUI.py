from misc import *
from ModelViewer import *
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf


class GUI(object):
    def __init__(self, models_list_generator, models_list_discriminator):

        self.n_col = 15
        self.n_row = 12
        self.image_size = 150

        self.default_value_k = 0
        self.default_value_mu = 0
        self.default_value_sigma = 1

        self.debounce_time = 50
        self.debounce_id_generator = None
        self.debounce_id_discriminator = None

        self.models_quantity = min(len(models_list_generator), len(models_list_discriminator))

        self.root = tk.Tk()
        self.root.configure(bg="black")
        self.root.title("GANalyzer")

        self.initializing = True

        self.initialize_input_panel()

        self.generator_viewer = ModelViewer(models_list_generator, 2, self.root, "Generator", 15, True, self)
        self.discriminator_viewer = ModelViewer(models_list_discriminator, 17, self.root, "Discriminator", 15, False, self)
        # self.initialize_generator_panel()
        # self.initialize_discriminator_panel()

        self.init_selectors()

        self.initializing = False

        self.update_generator()

        self.root.mainloop()

    def init_selectors(self):

        self.refresh_label_k(self.default_value_k)
        self.refresh_label_mu(self.default_value_mu)
        self.refresh_label_sigma(self.default_value_sigma)

        self.randomize_all_sliders(self.default_value_mu, self.default_value_sigma)

    def initialize_input_panel(self):
        title_input_hint = tk.Label(self.root, text="Input", bg="#666666")
        title_input_hint.grid(row=0, column=0, columnspan=self.n_col, rowspan=1, sticky="we")

        self.input_image_grid_size = int(latent_dimension_generator**0.5)

        self.max_slider_value = 5
        self.slider_width = 2 * self.max_slider_value

        notebook = ttk.Notebook(self.root)
        notebook.grid(row=1, column=0, columnspan=self.n_col, rowspan=1, sticky="nsew")

        # First tab
        random_input_tab = ttk.Frame(notebook)
        notebook.add(random_input_tab, text="Set Random Inputs")
        random_input_tab.columnconfigure((0, 1, 2, 3, 4, 5), weight=1)
        random_input_tab.rowconfigure((0, 1, 2, 3), weight=1)

        # Second tab
        constant_input_tab = ttk.Frame(notebook)
        notebook.add(constant_input_tab, text="Set Constant Input")
        constant_input_tab.columnconfigure((0, 1, 2, 3, 4, 5), weight=1)
        constant_input_tab.rowconfigure((0, 1, 2, 3), weight=1)

        # Third tab
        manual_input_tab = ttk.Frame(notebook)
        notebook.add(manual_input_tab, text="Set Input Manually")
        manual_input_tab.columnconfigure(tuple(range(self.input_image_grid_size)), weight=1)
        manual_input_tab.rowconfigure(tuple(range(self.input_image_grid_size)), weight=1)

        # Fill the tabs

        # Grid of sliders
        self.slider_grid = [[None for _ in range(self.input_image_grid_size)] for _ in range(self.input_image_grid_size)]
        for i in range(self.input_image_grid_size):
            for j in range(self.input_image_grid_size):
                self.slider_grid[i][j] = self.get_grid_slider(i, j, manual_input_tab)

        label_hint_constant = tk.Label(constant_input_tab, text="Set All Constant Value : ")
        label_hint_constant.grid(row=0, column=1, columnspan=2, sticky="nsew")

        label_hint_random = tk.Label(random_input_tab, text="Set All Random Value ")
        label_hint_random.grid(row=0, column=1, columnspan=2, sticky="nsew")

        self.label_k = self.create_parameter_input_label(1, 1, constant_input_tab)

        self.label_mu = self.create_parameter_input_label(1, 1, random_input_tab)
        self.label_sigma = self.create_parameter_input_label(1, 2, random_input_tab)

        self.slider_k = self.create_parameter_input_slider(self.default_value_k, 3, 1, True, self.refresh_label_k, constant_input_tab)

        self.slider_mu = self.create_parameter_input_slider(self.default_value_mu, 3, 1, True, self.refresh_label_mu, random_input_tab)
        self.slider_sigma = self.create_parameter_input_slider(self.default_value_sigma, 3, 2, False, self.refresh_label_sigma, random_input_tab)

        button_set_input_constant = ttk.Button(constant_input_tab, text="Set", command=self.set_input_constant)
        button_set_input_constant.grid(row=3, column=1, columnspan=2)

        button_set_input_random = ttk.Button(random_input_tab, text="Set", command=self.set_input_random)
        button_set_input_random.grid(row=3, column=1, columnspan=2)

    def get_grid_slider(self, i, j, parent):
        slider = ttk.Scale(parent, from_=-self.max_slider_value, to=self.max_slider_value, orient="horizontal", length=100)
        slider.grid(row=i + 1, column=j, padx=3, pady=3)
        slider.bind("<ButtonRelease-1>", self.update_generator)
        return slider

    def generate_image_from_input_values(self, input_raw):
        self.generator_viewer.current_input = np.array([input_raw])

        if rgb_images:
            predicted_raw = self.generator_viewer.current_model.predict(self.generator_viewer.current_input)[0, :, :, :]
        else:
            predicted_raw = self.generator_viewer.current_model.predict(self.generator_viewer.current_input)[0, :, :, 0]

        return find_limits_and_project(predicted_raw)

    def update_generator(self, event=None):
        if not self.initializing:  # TODO maybe put that if before method call ?

            values = [self.slider_grid[i][j].get() for i in range(self.input_image_grid_size) for j in range(self.input_image_grid_size)]

            # update image_in_generator
            self.refresh_image_in_generator(values)
            # self.generator_viewer.refresh_data_in(values)

            # update image_inside_generator
            # self.generator_viewer.refresh_inside_visualization("generator")

            # update image_out_generator
            self.refresh_image_out_generator(values)
            # self.generator_viewer.refresh_data_out(values)

            # Update discriminator
            self.update_discriminator()

    def update_discriminator(self):
        if not self.initializing:  # TODO maybe put that if before method call ?
            # update image_in_discriminator
            self.refresh_image_in_discriminator()

            # update image_inside_generator
            # self.refresh_inside_visualization("discriminator")
            # self.discriminator_viewer.refresh_inside_visualization("discriminator")

            # update prediction discriminator
            self.refresh_prediction_discriminator()

    def refresh_image_in_generator(self, values):
        input_before_reshape = np.array(values).reshape((self.input_image_grid_size, self.input_image_grid_size))
        input_after_reshape = project_array(input_before_reshape, 254, -self.max_slider_value, self.max_slider_value).astype(np.uint8)

        self.generator_viewer.refresh_tk_image(input_after_reshape, False, self.generator_viewer.data_in)

    def refresh_image_out_generator(self, values):
        self.generated_image = self.generate_image_from_input_values(values)
        self.generator_viewer.refresh_tk_image(self.generated_image, rgb_images, self.generator_viewer.data_out)

    def refresh_image_in_discriminator(self):
        self.discriminator_viewer.refresh_tk_image(self.generated_image, rgb_images, self.discriminator_viewer.data_in)

    def refresh_prediction_discriminator(self):
        self.discriminator_viewer.current_input = np.array([((self.generated_image - 127.5) / 127.5).astype(np.float64)])
        predicted_output = "ok"
        if model_name == "test_0" or model_name == "test_0B":
            predicted_output = self.discriminator_viewer.current_model.predict(self.discriminator_viewer.current_input)[0][0]
        elif model_name == "test_1":
            predicted_output = self.discriminator_viewer.current_model.predict(self.discriminator_viewer.current_input)[0][0][0][0]
        # self.label_prediction_out_discriminator.config(text="Prediction : " + str(round(predicted_output, 2)))
        self.discriminator_viewer.data_out.config(text="Prediction : " + str(round(predicted_output, 2)))

    def randomize_all_sliders(self, mu, sigma):
        for i in range(self.input_image_grid_size):
            for j in range(self.input_image_grid_size):
                random_value = np.random.normal(loc=mu, scale=sigma)
                val_clipped = max(-self.max_slider_value, min(self.max_slider_value, random_value))  # clip value

                self.slider_grid[i][j].set(val_clipped)

        self.update_generator()

    def set_input_constant(self):
        new_k_value = self.slider_k.get()
        for i in range(self.input_image_grid_size):
            for j in range(self.input_image_grid_size):
                self.slider_grid[i][j].set(new_k_value)

        self.update_generator()

    def set_input_random(self):
        new_mu_value = self.slider_mu.get()
        new_sigma_value = self.slider_sigma.get()
        self.randomize_all_sliders(new_mu_value, new_sigma_value)

    def create_parameter_input_label(self, x, y, parent):
        label = tk.Label(parent)
        label.grid(row=y, column=x, columnspan=2)
        return label

    def create_parameter_input_slider(self, default_value, x, y, can_be_negative, method_refresh_text, parent):
        if can_be_negative:
            this_min_value = -self.max_slider_value
        else:
            this_min_value = 0
        slider = ttk.Scale(parent, from_=this_min_value, to=self.max_slider_value, orient="horizontal", length=100, command=method_refresh_text)

        slider.grid(row=y, column=x)
        slider.set(default_value)

        return slider

    def refresh_label_k(self, event):
        self.label_k.config(text="K = " + str(round(float(event), 2)))

    def refresh_label_mu(self, event):
        self.label_mu.config(text="Mu = " + str(round(float(event), 2)))

    def refresh_label_sigma(self, event):
        self.label_sigma.config(text="Sigma = " + str(round(float(event), 2)))
