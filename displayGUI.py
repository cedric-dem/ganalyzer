from misc import *

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

class GUI(object):
    def __init__(self, models_list_generator, models_list_discriminator):

        self.image_size=150

        self.default_value_k = 0
        self.default_value_mu = 0
        self.default_value_sigma = 1

        self.models_quantity=min(len(models_list_generator),len(models_list_discriminator))
        self.generator=None
        self.discriminator = None
        self.models_list_generator=models_list_generator
        self.models_list_discriminator=models_list_discriminator

        self.root = tk.Tk()
        self.root.configure(bg="black")
        self.root.title("GANalyzer")


        self.initializing=True

        self.initialize_input_panel()
        self.initialize_generator_panel()
        self.initialize_discriminator_panel()

        self.init_sliders()

        self.initializing=False

        self.update_generator()

        self.root.mainloop()

    def init_sliders(self):
        self.epoch_slider_generator.set(self.models_quantity - 1)
        self.epoch_slider_discriminator.set(self.models_quantity - 1)

        self.on_generator_epoch_slider_change(self.models_quantity - 1)
        self.on_discriminator_epoch_slider_change(self.models_quantity - 1)

        self.refresh_label_k(self.default_value_k)
        self.refresh_label_mu(self.default_value_mu)
        self.refresh_label_sigma(self.default_value_sigma)

        self.randomize_all_sliders(self.default_value_mu, self.default_value_sigma)

    def initialize_input_panel(self):
        input_data_panel_hint = tk.Label(self.root,text="Set Input Data", bg="#666666")
        input_data_panel_hint.grid(row=0, column=0, columnspan=15, pady=10,sticky='we')

        self.grid_size = int(latent_dimension_generator ** 0.5)

        self.max_slider_value = 5
        self.slider_width = 2 * self.max_slider_value

        # Grid of sliders
        self.slider_grid = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.slider_grid[i][j] = self.get_grid_slider(i, j)

        hint_constant = tk.Label(self.root, text="Set All Constant Value : ")
        hint_constant.grid(row=self.grid_size+1, column=0, columnspan=2, pady=10)

        hint_random = tk.Label(self.root, text="Set All Random Value ")
        hint_random.grid(row=self.grid_size + 2, column=0, columnspan=2, pady=10)

        self.k_label = self.create_parameter_input_label(self.root, 4, self.grid_size + 1)
        self.mu_label = self.create_parameter_input_label(self.root, 2, self.grid_size + 2)
        self.sigma_label = self.create_parameter_input_label(self.root, 5, self.grid_size + 2)

        self.k_slider = self.create_parameter_input_slider(self.root,  self.default_value_k, 4, self.grid_size +1, True, self.refresh_label_k)
        self.mu_slider = self.create_parameter_input_slider(self.root, self.default_value_mu, 2, self.grid_size + 2, True, self.refresh_label_mu)
        self.sigma_slider = self.create_parameter_input_slider(self.root, self.default_value_sigma, 5, self.grid_size + 2, False, self.refresh_label_sigma)

        btn_set_input_constant = ttk.Button(self.root, text="Set", command=self.set_input_constant)
        btn_set_input_constant.grid(row=self.grid_size+1, column=7, columnspan=2, pady=10)

        btn_set_input_random = ttk.Button(self.root, text="Set", command=self.set_input_random)
        btn_set_input_random.grid(row=self.grid_size + 2, column=7, columnspan=2, pady=10)

    def get_grid_slider(self, i, j):
        slider = ttk.Scale(self.root, from_=-self.max_slider_value, to=self.max_slider_value, orient='horizontal',
                           length=100)
        slider.grid(row=i + 1, column=j, padx=3, pady=3)
        slider.bind("<ButtonRelease-1>", self.update_generator)
        return slider

    def initialize_generator_panel(self):
        self.input_panel_height = self.grid_size + 3

        generator_hint = tk.Label(self.root,text="Generator", bg="#666666")
        generator_hint.grid(row=self.input_panel_height, column=0, columnspan=15, pady=10,sticky='we')

        self.current_epoch_generator_text = tk.Label(self.root)
        self.current_epoch_generator_text.grid(row=self.input_panel_height + 4, column=0, columnspan=2, pady=10)

        #TODO : convert if possible to on click release to avoid computation
        self.epoch_slider_generator = ttk.Scale(self.root, from_=0, to=self.models_quantity - 1, orient='horizontal', length=600, command=self.on_generator_epoch_slider_change)
        self.epoch_slider_generator.grid(row=self.input_panel_height + 5, column=0, columnspan=self.grid_size - 6, padx=10, pady=20, sticky='ew')

        self.image_in_generator = tk.Label(self.root)
        self.image_in_generator.grid(row=self.input_panel_height+1, column=self.grid_size - 2, rowspan=self.grid_size + 1, padx=20, pady=10)

        self.image_out_generator = tk.Label(self.root)
        self.image_out_generator.grid(row=self.input_panel_height+1, column=self.grid_size, rowspan=self.grid_size + 1, padx=20, pady=10)

    def initialize_discriminator_panel(self):
        self.input_and_generator_panel_height = self.input_panel_height + 15

        discriminator_hint = tk.Label(self.root,text="Discriminator", bg="#666666")
        discriminator_hint.grid(row=self.input_and_generator_panel_height, column=0, columnspan=15, pady=10,sticky='we')

        self.current_epoch_discriminator_text = tk.Label(self.root)
        self.current_epoch_discriminator_text.grid(row=self.input_and_generator_panel_height + 4, column=0, columnspan=2, pady=10)

        #TODO : convert if possible to on click release to avoid computation
        self.epoch_slider_discriminator = ttk.Scale(self.root, from_=0, to=self.models_quantity - 1, orient='horizontal', length=600, command=self.on_discriminator_epoch_slider_change)
        self.epoch_slider_discriminator.grid(row=self.input_and_generator_panel_height + 5, column=0, columnspan=self.grid_size - 6, padx=10, pady=20, sticky='ew')

        self.image_in_discriminator = tk.Label(self.root)
        self.image_in_discriminator.grid(row=self.input_and_generator_panel_height+1, column=self.grid_size - 2, rowspan=self.grid_size + 1, padx=20, pady=10)

        self.prediction_out_discriminator = tk.Label(self.root)
        self.prediction_out_discriminator.grid(row=self.input_and_generator_panel_height + 1, column=self.grid_size, rowspan=self.grid_size + 1, padx=20, pady=10,sticky='nsew')

    def generate_image_from_input_values(self, input_raw):
        input_rebound = np.array([input_raw]) / self.slider_width

        if rgb_images:
            predicted_raw = self.generator.predict(input_rebound)[0, :, :, :]
        else:
            predicted_raw = self.generator.predict(input_rebound)[0, :, :, 0]

        return  set_interval(predicted_raw)

    def update_generator(self, event= None):
        if not self.initializing: #TODO maybe put that if before method call ?
            print('==> Update Generator')

            values = [self.slider_grid[i][j].get() for i in range(self.grid_size) for j in range(self.grid_size)]

            #update image_in_generator
            self.refresh_image_in_generator(values)

            #update image_out_generator
            self.refresh_image_out_generator(values)

            # Update discriminator
            self.update_discriminator()

    def refresh_image_in_generator(self, values):
        input_before_reshape = np.array(values).reshape((self.grid_size, self.grid_size))
        input_after_reshape = project_array(input_before_reshape, 254, -self.max_slider_value,
                                            self.max_slider_value).astype(np.uint8)
        img = Image.fromarray(input_after_reshape, mode='L')
        img_tk = ImageTk.PhotoImage(img.resize((self.image_size, self.image_size), Image.NEAREST))
        self.image_in_generator.configure(image=img_tk)
        self.image_in_generator.image = img_tk

    def refresh_image_out_generator(self, values):
        self.generated_image = self.generate_image_from_input_values(values)
        if rgb_images:
            img = Image.fromarray(self.generated_image.astype('uint8'), mode='RGB')
        else:
            img = Image.fromarray(self.generated_image, mode='L')
        img_tk = ImageTk.PhotoImage(img.resize((self.image_size, self.image_size), Image.NEAREST))
        self.image_out_generator.configure(image=img_tk)
        self.image_out_generator.image = img_tk

    def update_discriminator(self):
        if not self.initializing: #TODO maybe put that if before method call ?
            print('==> Update Discriminator')

            #update image_in_discriminator
            self.refresh_image_in_discriminator()

            #update prediction discriminator
            self.refresh_prediction_discriminator()

    def refresh_image_in_discriminator(self):
        if rgb_images:
            img = Image.fromarray(self.generated_image.astype('uint8'), mode='RGB')
        else:
            img = Image.fromarray(self.generated_image, mode='L')
        img_tk = ImageTk.PhotoImage(img.resize((self.image_size, self.image_size), Image.NEAREST))
        self.image_in_discriminator.configure(image=img_tk)
        self.image_in_discriminator.image = img_tk

    def refresh_prediction_discriminator(self):
        input_image_discriminator = np.array([((self.generated_image - 127.5) / 127.5).astype(np.float64)])
        predicted_output = self.discriminator.predict(input_image_discriminator)[0][0][0][0]
        self.prediction_out_discriminator.config(text="Prediction : " + str(round(predicted_output, 2)))

    def randomize_all_sliders(self, mu, sigma):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                val = np.random.normal(loc=mu, scale=sigma)
                val_clipped = max(-self.max_slider_value, min(self.max_slider_value, val))  # clip entre 0 et 1

                self.slider_grid[i][j].set(val_clipped)

        self.update_generator()

    def on_generator_epoch_slider_change(self, value):
        #new_epoch=int(float(self.epoch_slider_generator.get()))
        new_epoch=int(float(value))
        self.generator=self.models_list_generator[new_epoch]

        self.update_generator()

        self.current_epoch_generator_text.config(text="Current Epoch : " + str(new_epoch) + " / " + str(self.models_quantity - 1))

    def on_discriminator_epoch_slider_change(self, value):
        #new_epoch=int(float(self.epoch_slider_discriminator.get()))
        new_epoch=int(float(value))
        self.discriminator=self.models_list_discriminator[new_epoch]
        self.update_discriminator()

        self.current_epoch_discriminator_text.config(text="Current Epoch : " + str(new_epoch) + " / " + str(self.models_quantity - 1))

    def set_input_constant(self):
        new_k_value=self.k_slider.get()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.slider_grid[i][j].set(new_k_value)

        self.update_generator()

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
generator_list = get_all_models("generator")
discriminators_list = get_all_models("discriminator")

print('==> Number of loaded models : ',len(generator_list))

main_gui=GUI(generator_list, discriminators_list)