import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class ModelViewer(object):
    def __init__(self, models_list):

        self.current_model = None
        self.models_list = models_list
