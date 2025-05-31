from misc import *

import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

import keras

def get_all_models():
    result=[]
    i=0
    this_filename=get_generator_model_path_at_given_epoch(0)
    while os.path.isfile(this_filename) and i<10:
        result.append(keras.models.load_model(this_filename))
        this_filename=get_generator_model_path_at_given_epoch(i)
        print("=> Attempt to load epoch ",i)
        i+=1
    return result

class GUI(object):
    def __init__(self, models_list):

        self.models_quantity=len(models_list)
        self.generator=None
        self.models_list=models_list

        self.root = tk.Tk()
        self.root.title("GANalyzer")

        self.initialize_input_panel()
        self.initialize_generator_panel()
        self.initialize_discriminator_panel()

        self.root.mainloop()

    def initialize_input_panel(self):
        default_value_k = 0
        default_value_mu = 0
        default_value_sigma = 1

        self.grid_size = int(latent_dimension_generator ** 0.5)

        self.max_slider_value = 5
        self.slider_width = 2 * self.max_slider_value

        # Grid of sliders
        self.slider_grid = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                slider = ttk.Scale(self.root, from_=-self.max_slider_value, to=self.max_slider_value, orient='horizontal', length=100)
                slider.grid(row=i, column=j, padx=3, pady=3)
                slider.bind("<ButtonRelease-1>", self.update_image)
                self.slider_grid[i][j] = slider

        hint_constant = tk.Label(self.root, text="Set All Constant Value : ")
        hint_constant.grid(row=self.grid_size, column=0, columnspan=2, pady=10)

        hint_random = tk.Label(self.root, text="Set All Random Value ")
        hint_random.grid(row=self.grid_size + 1, column=0, columnspan=2, pady=10)

        self.k_label = self.create_parameter_input_label(self.root, 4, self.grid_size)
        self.mu_label = self.create_parameter_input_label(self.root, 2, self.grid_size + 1)
        self.sigma_label = self.create_parameter_input_label(self.root, 5, self.grid_size + 1)

        self.k_slider = self.create_parameter_input_slider(self.root,  default_value_k, 4, self.grid_size, True, self.refresh_label_k)
        self.mu_slider = self.create_parameter_input_slider(self.root, default_value_mu, 2, self.grid_size + 1, True, self.refresh_label_mu)
        self.sigma_slider = self.create_parameter_input_slider(self.root, default_value_sigma, 5, self.grid_size + 1, False, self.refresh_label_sigma)

        btn_set_input_constant = ttk.Button(self.root, text="Set", command=self.set_input_constant)
        btn_set_input_constant.grid(row=self.grid_size, column=7, columnspan=2, pady=10)

        btn_set_input_random = ttk.Button(self.root, text="Set", command=self.set_input_random)
        btn_set_input_random.grid(row=self.grid_size + 1, column=7, columnspan=2, pady=10)

        # Image on the right
        self.image_label = tk.Label(self.root)
        self.image_label.grid(row=0, column=self.grid_size, rowspan=self.grid_size + 1, padx=20, pady=10)

        self.current_epoch_text = tk.Label(self.root)
        self.current_epoch_text.grid(row=self.grid_size + 2, column=0, columnspan=2, pady=10)

        time_slider = ttk.Scale(self.root, from_=0, to=self.models_quantity - 1, orient='horizontal', length=600, command=self.on_epoch_slider_change)
        time_slider.grid(row=self.grid_size + 2, column=3, columnspan=self.grid_size - 3, padx=10, pady=20, sticky='ew')

        time_slider.set(self.models_quantity - 1)
        self.on_epoch_slider_change(self.models_quantity - 1)

        self.refresh_label_k(default_value_k)
        self.refresh_label_mu(default_value_mu)
        self.refresh_label_sigma(default_value_sigma)

        self.randomize_all_sliders(default_value_mu, default_value_sigma)

    def initialize_generator_panel(self):
        pass

    def initialize_discriminator_panel(self):
        pass

    def generate_image_from_input_values(self, input_raw):
        input_rebound = np.array([input_raw]) / self.slider_width

        if rgb_images:
            predicted_raw = self.generator.predict(input_rebound)[0, :, :, :]
        else:
            predicted_raw = self.generator.predict(input_rebound)[0, :, :, 0]

        return  self.set_interval(predicted_raw)

    @staticmethod
    def set_interval(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)

        delta = max_val - min_val
        if delta>0:
            projected = (arr - min_val) / delta * 254
        else:
            projected = arr

        return np.round(projected).astype(np.uint8)

    def update_image(self, event= None):
        values = [self.slider_grid[i][j].get() for i in range(self.grid_size) for j in range(self.grid_size)]
        img_array = self.generate_image_from_input_values(values)
        if rgb_images:
            img = Image.fromarray(img_array.astype('uint8'), mode='RGB')
        else:
            img = Image.fromarray(img_array, mode='L')
        img_tk = ImageTk.PhotoImage(img.resize((140, 140), Image.NEAREST))
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

    def randomize_all_sliders(self, mu, sigma):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                val = np.random.normal(loc=mu, scale=sigma)
                val_clipped = max(-self.max_slider_value, min(self.max_slider_value, val))  # clip entre 0 et 1

                self.slider_grid[i][j].set(val_clipped)

        self.update_image()

    def on_epoch_slider_change(self, value):
        new_epoch=int(float(value))
        self.generator=self.models_list[new_epoch]
        self.update_image()

        self.current_epoch_text.config(text="Current Epoch : "+str(new_epoch)+" / "+str(self.models_quantity - 1))

    def set_input_constant(self):
        new_k_value=self.k_slider.get()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.slider_grid[i][j].set(new_k_value)
        self.update_image()

    def set_input_random(self):
        new_mu_value=self.mu_slider.get()
        new_sigma_value=self.sigma_slider.get()
        self.randomize_all_sliders(new_mu_value, new_sigma_value)

    @staticmethod
    def create_parameter_input_label(root, x, y):
        label = tk.Label(root)
        label.grid(row=y, column=x-1, columnspan=2, pady=10)
        return label

    def create_parameter_input_slider(self, root, default_value, x, y, can_be_negative, method_refresh_text):
        if can_be_negative:
            this_min_value=-self.max_slider_value
        else:
            this_min_value=0
        slider = ttk.Scale(root, from_=this_min_value, to=self.max_slider_value, orient='horizontal', length=100, command= method_refresh_text)

        slider.grid(row=y, column=x+1, padx=3, pady=3)
        slider.set(default_value)

        return slider

    def refresh_label_k(self, event):
        self.k_label.config(text="K = "+ str(round(float(event), 2)))

    def refresh_label_mu(self, event):
        self.mu_label.config(text="Mu = "+ str(round(float(event), 2)))

    def refresh_label_sigma(self, event):
        self.sigma_label.config(text="Sigma = "+ str(round(float(event), 2)))

# load all the models
allModels = get_all_models()
print('==> Number of loaded models : ',len(allModels))
main_gui=GUI(allModels)