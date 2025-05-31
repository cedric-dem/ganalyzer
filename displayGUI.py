from config import *

import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

import keras

def get_all_models():
    result=[]
    i=0
    again=True
    while again:
        print("=> Attempt to load epoch ",i)

        this_filename= model_path+'generator_epoch_' + str(i) + ".keras"

        again=os.path.isfile(this_filename)

        if again:
            result.append(keras.models.load_model(this_filename))

        i+=1

    return result

def generate_image_from_input_values(input_raw):
    global max_magn_twice
    input_rebound = np.array([input_raw]) / max_magn_twice

    if rgb_images:
        predicted_raw = generator.predict(input_rebound)[0, :, :, :]
    else:

        predicted_raw = generator.predict(input_rebound)[0, :, :, 0]
    predicted_rebound = set_interval(predicted_raw)

    return predicted_rebound

def set_interval(arr):
	min_val = np.min(arr)
	max_val = np.max(arr)

	projected = (arr - min_val) / (max_val - min_val) * 254
	return np.round(projected).astype(np.uint8)

def update_image(event= None):
    global slider_grid
    global image_label
    values = [slider_grid[i][j].get() for i in range(n) for j in range(n)]
    img_array = generate_image_from_input_values(values)
    if rgb_images:
        img = Image.fromarray(img_array.astype('uint8'), mode='RGB')
    else:
        img = Image.fromarray(img_array, mode='L')
    img_tk = ImageTk.PhotoImage(img.resize((140, 140), Image.NEAREST))
    image_label.configure(image=img_tk)
    image_label.image = img_tk

def randomize_sliders_low_variance():
    randomize_sliders_with_given_sigma(1)

def randomize_sliders_high_variance():
    randomize_sliders_with_given_sigma(2)

def randomize_sliders_with_given_sigma(sigma):
    global max_magn, n
    for i in range(n):
        for j in range(n):
            val = np.random.normal(loc=0.0, scale=sigma)
            val_clipped = max(-max_magn, min(max_magn, val))  # clip entre 0 et 1

            slider_grid[i][j].set(val_clipped)

    update_image()

def set_sliders_low():
    for i in range(n):
        for j in range(n):
            slider_grid[i][j].set(-max_magn)
    update_image()

def set_sliders_zero():
    for i in range(n):
        for j in range(n):
            slider_grid[i][j].set(0)
    update_image()

def set_sliders_high():
    for i in range(n):
        for j in range(n):
            slider_grid[i][j].set(max_magn)
    update_image()

def on_epoch_slider_change(value):
    global generator
    generator=allModels[int(float(value))]
    update_image()

def initialize_gui():
    global slider_grid,n
    n=int(latent_dimension_generator**0.5)
    global max_magn
    max_magn=5

    global max_magn_twice
    max_magn_twice=2*max_magn

    root = tk.Tk()
    root.title(f"Grid {n}x{n} sliders")

    # Grille de sliders
    slider_grid = [[None for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i * j <= latent_dimension_generator:
                slider = ttk.Scale(root, from_=-max_magn, to=max_magn, orient='horizontal', length=100)
                slider.grid(row=i, column=j, padx=3, pady=3)
                slider.bind("<ButtonRelease-1>", update_image)
                slider_grid[i][j] = slider

    # Bouton random
    set_min_button = ttk.Button(root, text="Set low", command=set_sliders_low)
    set_min_button.grid(row=n, column=0, columnspan=2, pady=10)

    very_random_button = ttk.Button(root, text="Set Very Random", command=randomize_sliders_high_variance)
    very_random_button.grid(row=n, column=1, columnspan=2, pady=10)

    random_button = ttk.Button(root, text="Set Random", command=randomize_sliders_low_variance)
    random_button.grid(row=n, column=2, columnspan=2, pady=10)

    zero_button = ttk.Button(root, text="Set Zero", command=set_sliders_zero)
    zero_button.grid(row=n, column=3, columnspan=2, pady=10)

    set_max_button = ttk.Button(root, text="Set high", command=set_sliders_high)
    set_max_button.grid(row=n, column=4, columnspan=2, pady=10)

    # Imag on the right
    global image_label
    image_label = tk.Label(root)
    image_label.grid(row=0, column=n, rowspan=n + 1, padx=20, pady=10)

    time_slider = ttk.Scale(root, from_=0, to=nmodels - 1, orient='horizontal', length=600, command=on_epoch_slider_change)
    time_slider.grid(row=n + 1, column=0, columnspan=n, padx=10, pady=20, sticky='ew')

    randomize_sliders_low_variance()

    root.mainloop()

# load all the models
allModels = get_all_models()
nmodels = len(allModels)
print('==> Number of loaded models : ',nmodels)
generator = allModels[0]

initialize_gui()