import time

from misc import *

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np


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
        self.generator = None
        self.discriminator = None
        self.models_list_generator = models_list_generator
        self.models_list_discriminator = models_list_discriminator

        self.root = tk.Tk()
        self.root.configure(bg="black")
        self.root.title("GANalyzer")

        self.initializing = True

        self.initialize_input_panel()
        self.initialize_generator_panel()
        self.initialize_discriminator_panel()

        self.init_sliders()

        self.initializing = False

        self.update_generator()

        self.root.mainloop()

    def init_sliders(self):
        self.slider_epoch_generator.set(self.models_quantity - 1)
        self.slider_epoch_discriminator.set(self.models_quantity - 1)

        self.on_generator_epoch_slider_change(self.models_quantity - 1)
        self.on_discriminator_epoch_slider_change(self.models_quantity - 1)

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

    def initialize_generator_panel(self):
        self.input_panel_height = 2

        title_generator_hint = tk.Label(self.root, text="Generator", bg="#666666")
        title_generator_hint.grid(row=self.input_panel_height, column=0, rowspan=1, columnspan=self.n_col, sticky="we")

        self.label_current_epoch_generator, self.slider_epoch_generator, self.image_in_generator, self.image_out_generator = self.create_model_panel(self.on_generator_epoch_slider_change_debounced)

    def initialize_discriminator_panel(self):
        self.input_and_generator_panel_height = self.input_panel_height + 15

        title_discriminator_hint = tk.Label(self.root, text="Discriminator", bg="#666666")
        title_discriminator_hint.grid(row=self.input_and_generator_panel_height, column=0, columnspan=15, sticky="we")

        self.label_current_epoch_discriminator, self.slider_epoch_discriminator, self.image_in_discriminator, self.label_prediction_out_discriminator = self.create_model_panel(self.on_discriminator_epoch_slider_change_debounced)

    def create_model_panel(self, cmd):
        layout_panel = tk.Frame(self.root, bg="#333333")
        layout_panel.grid(rowspan=1, columnspan=self.n_col, sticky="nsew")
        layout_panel.columnconfigure((0, 1, 2, 3, 4, 5), weight=1)
        layout_panel.rowconfigure((0), weight=1)

        layout_epoch = tk.Frame(layout_panel, bg="#111111")
        layout_epoch.grid(rowspan=1, columnspan=1, sticky="nsew")
        layout_epoch.columnconfigure((0), weight=1)
        layout_epoch.rowconfigure((0,1), weight=1)
        label_epoch = tk.Label(layout_epoch)
        label_epoch.grid(row=0, column=0, columnspan=1)
        slider_epoch = ttk.Scale(layout_epoch, from_=0, to=self.models_quantity - 1, orient="horizontal", command=cmd)
        slider_epoch.grid(row=1, column=0, columnspan=1, sticky="ew")

        model_in = tk.Label(layout_panel)
        model_in.grid(row=0, column=2, rowspan=1, columnspan=1)

        model_out = tk.Label(layout_panel)
        model_out.grid(row=0, column=3, rowspan=1, columnspan=1, sticky="nsew")

        return label_epoch, slider_epoch, model_in, model_out

    def generate_image_from_input_values(self, input_raw):
        input_rebound = np.array([input_raw]) / self.slider_width

        if rgb_images:
            predicted_raw = self.generator.predict(input_rebound)[0, :, :, :]
        else:
            predicted_raw = self.generator.predict(input_rebound)[0, :, :, 0]

        return find_limits_and_project(predicted_raw)

    def update_generator(self, event=None):
        if not self.initializing:  # TODO maybe put that if before method call ?

            values = [self.slider_grid[i][j].get() for i in range(self.input_image_grid_size) for j in range(self.input_image_grid_size)]

            # update image_in_generator
            self.refresh_image_in_generator(values)

            # update image_out_generator
            self.refresh_image_out_generator(values)

            # Update discriminator
            self.update_discriminator()

    def refresh_image_in_generator(self, values):
        input_before_reshape = np.array(values).reshape((self.input_image_grid_size, self.input_image_grid_size))
        input_after_reshape = project_array(input_before_reshape, 254, -self.max_slider_value, self.max_slider_value).astype(np.uint8)

        self.refresh_tk_image(input_after_reshape, False, self.image_in_generator)

    def refresh_tk_image(self, input_matrix, is_color, tk_image: tk.Label):
        if is_color:
            img = Image.fromarray(input_matrix.astype("uint8"), mode="RGB")
        else:
            img = Image.fromarray(input_matrix, mode="L")

        img_tk = ImageTk.PhotoImage(img.resize((self.image_size, self.image_size), Image.NEAREST))
        tk_image.configure(image=img_tk)
        tk_image.image = img_tk

    def refresh_image_out_generator(self, values):
        self.generated_image = self.generate_image_from_input_values(values)
        self.refresh_tk_image(self.generated_image, rgb_images, self.image_out_generator)

    def update_discriminator(self):
        if not self.initializing:  # TODO maybe put that if before method call ?
            # update image_in_discriminator
            self.refresh_image_in_discriminator()

            # update prediction discriminator
            self.refresh_prediction_discriminator()

    def refresh_image_in_discriminator(self):
        self.refresh_tk_image(self.generated_image, rgb_images, self.image_in_discriminator)

    def refresh_prediction_discriminator(self):
        input_image_discriminator = np.array([((self.generated_image - 127.5) / 127.5).astype(np.float64)])
        if model_name == "test_0" or model_name == "test_0B":
            predicted_output = self.discriminator.predict(input_image_discriminator)[0][0]
        elif model_name == "test_1":
            predicted_output = self.discriminator.predict(input_image_discriminator)[0][0][0][0]
        self.label_prediction_out_discriminator.config(text="Prediction : " + str(round(predicted_output, 2)))

    def randomize_all_sliders(self, mu, sigma):
        for i in range(self.input_image_grid_size):
            for j in range(self.input_image_grid_size):
                random_value = np.random.normal(loc=mu, scale=sigma)
                val_clipped = max(-self.max_slider_value, min(self.max_slider_value, random_value))  # clip value

                self.slider_grid[i][j].set(val_clipped)

        self.update_generator()

    def on_generator_epoch_slider_change_debounced(self, value):
        if self.debounce_id_generator:
            self.root.after_cancel(self.debounce_id_generator)

        # Schedule the real handler to run after debounce_time
        self.debounce_id_generator = self.root.after(self.debounce_time, self.on_generator_epoch_slider_change, value)

    def on_generator_epoch_slider_change(self, value):
        # new_epoch=int(float(self.epoch_slider_generator.get()))
        new_epoch = int(float(value))
        self.generator = self.models_list_generator[new_epoch]

        self.update_generator()

        self.label_current_epoch_generator.config(text="Current Epoch : " + str(new_epoch) + " / " + str(self.models_quantity - 1))

    def on_discriminator_epoch_slider_change_debounced(self, value):
        if self.debounce_id_discriminator:
            self.root.after_cancel(self.debounce_id_discriminator)

        # Schedule the real handler to run after debounce_time
        self.debounce_id_discriminator = self.root.after(self.debounce_time, self.on_discriminator_epoch_slider_change, value)

    def on_discriminator_epoch_slider_change(self, value):
        # new_epoch=int(float(self.epoch_slider_discriminator.get()))
        new_epoch = int(float(value))
        self.discriminator = self.models_list_discriminator[new_epoch]
        self.update_discriminator()

        self.label_current_epoch_discriminator.config(text="Current Epoch : " + str(new_epoch) + " / " + str(self.models_quantity - 1))

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


# load all the models
t0 = time.time()
generator_list = get_all_models("generator")
discriminators_list = get_all_models("discriminator")
t1 = time.time()
print("==> Time taken to load : ", round(t1 - t0, 2))

print("==> Number of loaded models : ", len(generator_list))

main_gui = GUI(generator_list, discriminators_list)
