import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class ModelViewer(object):
    def __init__(self, models_list):

        self.current_model = None
        self.models_list = models_list

    def get_layers_list(self):
        return [str(i) + ") " + self.current_model.layers[i].name for i in range(len(self.current_model.layers))]

    def get_layer_index(self, layer_name):
        return int(layer_name.split(")")[0])
