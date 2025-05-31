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
    global slider_width
    input_rebound = np.array([input_raw]) / slider_width

    if rgb_images:
        predicted_raw = generator.predict(input_rebound)[0, :, :, :]
    else:

        predicted_raw = generator.predict(input_rebound)[0, :, :, 0]
    predicted_rebound = set_interval(predicted_raw)

    return predicted_rebound

def set_interval(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)

    delta = max_val - min_val
    if delta>0:
        projected = (arr - min_val) / (delta) * 254
    else:
        projected = arr

    return np.round(projected).astype(np.uint8)

def update_image(event= None):
    global slider_grid, image_label
    values = [slider_grid[i][j].get() for i in range(grid_size) for j in range(grid_size)]
    img_array = generate_image_from_input_values(values)
    if rgb_images:
        img = Image.fromarray(img_array.astype('uint8'), mode='RGB')
    else:
        img = Image.fromarray(img_array, mode='L')
    img_tk = ImageTk.PhotoImage(img.resize((140, 140), Image.NEAREST))
    image_label.configure(image=img_tk)
    image_label.image = img_tk

def randomize_sliders_with_given_sigma(mu, sigma):
    global max_slider_value, grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            val = np.random.normal(loc=mu, scale=sigma)
            val_clipped = max(-max_slider_value, min(max_slider_value, val))  # clip entre 0 et 1

            slider_grid[i][j].set(val_clipped)

    update_image()

def on_epoch_slider_change(value):
    global generator, current_epoch_text

    new_epoch=int(float(value))
    generator=allModels[new_epoch]
    update_image()

    current_epoch_text.config(text="Current Epoch : "+str(new_epoch)+" / "+str(nmodels-1))

def set_input_constant():
    global k_slider
    new_k_value=k_slider.get()
    for i in range(grid_size):
        for j in range(grid_size):
            slider_grid[i][j].set(new_k_value)
    update_image()

def set_input_random():
    global mu_slider, sigma_slider
    new_mu_value=mu_slider.get()
    new_sigma_value=sigma_slider.get()
    randomize_sliders_with_given_sigma(new_mu_value, new_sigma_value)

def create_parameter_input_slider(root, name, default_value, x, y, can_be_negative):
    label = tk.Label(root, text=name+" = "+str(float(default_value)))
    label.grid(row=y, column=x-1, columnspan=2, pady=10)
    if can_be_negative:
        slider = ttk.Scale(root, from_=-max_slider_value, to=max_slider_value, orient='horizontal', length=100)
    else:
        slider = ttk.Scale(root, from_=0, to=max_slider_value, orient='horizontal', length=100)
    slider.grid(row=y, column=x+1, padx=3, pady=3)
    return slider

def initialize_gui():
    global grid_size, max_slider_value, slider_width, slider_grid, k_slider, mu_slider, sigma_slider, image_label, current_epoch_text
    grid_size=int(latent_dimension_generator ** 0.5)

    max_slider_value=5
    slider_width= 2 * max_slider_value

    root = tk.Tk()
    root.title("GANalyzer")

    # Grille de sliders
    slider_grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]
    for i in range(grid_size):
        for j in range(grid_size):
            if i * j <= latent_dimension_generator:
                slider = ttk.Scale(root, from_=-max_slider_value, to=max_slider_value, orient='horizontal', length=100)
                slider.grid(row=i, column=j, padx=3, pady=3)
                slider.bind("<ButtonRelease-1>", update_image)
                slider_grid[i][j] = slider

    hint_constant = tk.Label(root, text="Set All Constant Value : ")
    hint_constant.grid(row=grid_size, column=0, columnspan=2, pady=10)

    hint_random = tk.Label(root, text="Set All Random Value ")
    hint_random.grid(row=grid_size+1, column=0, columnspan=2, pady=10)

    k_slider=create_parameter_input_slider(root,"k",0, 4, grid_size, True)
    mu_slider=create_parameter_input_slider(root,"mu",0, 2, grid_size+1, True)
    sigma_slider=create_parameter_input_slider(root,"sigma",1, 5, grid_size+1,False)

    btn_set_input_constant = ttk.Button(root, text="Set", command=set_input_constant)
    btn_set_input_constant.grid(row=grid_size, column=7, columnspan=2, pady=10)

    btn_set_input_random = ttk.Button(root, text="Set", command=set_input_random)
    btn_set_input_random.grid(row=grid_size+1, column=7, columnspan=2, pady=10)

    # Image on the right
    image_label = tk.Label(root)
    image_label.grid(row=0, column=grid_size, rowspan=grid_size + 1, padx=20, pady=10)

    current_epoch_text = tk.Label(root)
    current_epoch_text.grid(row=grid_size+2, column=0, columnspan=2, pady=10)

    time_slider = ttk.Scale(root, from_=0, to=nmodels - 1, orient='horizontal', length=600, command=on_epoch_slider_change)
    time_slider.grid(row=grid_size + 2, column=3, columnspan=grid_size-3, padx=10, pady=20, sticky='ew')

    time_slider.set(nmodels-1)
    on_epoch_slider_change(nmodels-1)

    randomize_sliders_with_given_sigma(0,1)

    root.mainloop()

global current_epoch_text,k_slider, mu_slider, sigma_slider,slider_width, slider_grid,grid_size, max_slider_value,image_label

# load all the models
allModels = get_all_models()
nmodels = len(allModels)
print('==> Number of loaded models : ',nmodels)
generator = allModels[0]

initialize_gui()