from config import *

import os
import tkinter as tk
from tkinter import ttk

import keras

def getAllModels():
    result=[]
    i=0
    again=True
    while again  and i<10:
        print("=> Attempt to load epoch ",i)

        this_filename= model_path+'generator_epoch_' + str(i) + ".keras"

        again=os.path.isfile(this_filename)

        if again:
            result.append(keras.models.load_model(this_filename))

        i+=1

    return result

def update_image():
    pass

def set_all_sliders_min():
    pass

def randomize_high_variance():
    pass

def randomize_all_sliders():
    pass

def set_all_sliders_max():
    pass

def on_time_slider_change(value):
    pass

def set_all_sliders_zer():
    pass

def initializeGUI():
    n=int(latent_dimension_generator**0.5)
    max_magn=5
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
    set_min_button = ttk.Button(root, text="Set low", command=set_all_sliders_min)
    set_min_button.grid(row=n, column=0, columnspan=2, pady=10)

    very_random_button = ttk.Button(root, text="Set Very Random", command=randomize_high_variance)
    very_random_button.grid(row=n, column=1, columnspan=2, pady=10)

    random_button = ttk.Button(root, text="Set Random", command=randomize_all_sliders)
    random_button.grid(row=n, column=2, columnspan=2, pady=10)

    zero_button = ttk.Button(root, text="Set Zero", command=set_all_sliders_zer)
    zero_button.grid(row=n, column=3, columnspan=2, pady=10)

    set_max_button = ttk.Button(root, text="Set high", command=set_all_sliders_max)
    set_max_button.grid(row=n, column=4, columnspan=2, pady=10)

    # Imag on the right
    image_label = tk.Label(root)
    image_label.grid(row=0, column=n, rowspan=n + 1, padx=20, pady=10)

    time_slider = ttk.Scale(root, from_=0, to=nmodels - 1, orient='horizontal', length=600, command=on_time_slider_change)
    time_slider.grid(row=n + 1, column=0, columnspan=n, padx=10, pady=20, sticky='ew')

    randomize_all_sliders()

    root.mainloop()

# load all the models
allModels = getAllModels()
nmodels = len(allModels)
print('==> Number of loaded models : ',nmodels)
generator = allModels[0]

initializeGUI()