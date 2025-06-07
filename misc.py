from config import *

import keras
import os
import numpy as np


def get_generator_model_path_at_given_epoch(i):
    return get_model_path_at_given_epoch("generator", i)


def get_discriminator_model_path_at_given_epoch(i):
    return get_model_path_at_given_epoch("discriminator", i)


def get_model_path_at_given_epoch(model_type, i):
    return model_path + model_type + "_epoch_" + str(i) + ".keras"


def get_all_models(model_type):
    models_quantity = get_current_epoch()

    result = [None for i in range (models_quantity)]

    load_every=models_quantity//show_models_gui

    for i in range (models_quantity):

        if (i==0 or i==models_quantity-1 or (i%load_every==0)):
            print("=> Attempt to load " + model_type + " epoch ", i)
            this_filename = get_model_path_at_given_epoch(model_type, i)
            result[i]=keras.models.load_model(this_filename)

        i += 1
    return result


def find_limits_and_project(arr):
    projected = project_array(arr, 254, np.min(arr), np.max(arr))
    return np.round(projected).astype(np.uint8)


def project_array(arr, to, project_from, project_to):
    delta = project_to - project_from
    if delta > 0:
        result = ((arr - project_from) / delta) * to
    else:
        result = arr
    return result


def get_number_of_existing_models(filename):
    current_i = 0
    while os.path.isfile(filename + str(current_i) + ".keras"):
        current_i += 1
    return current_i - 1


def get_current_epoch():
    counter_generator = get_number_of_existing_models(model_path + "generator_epoch_")
    counter_discriminator = get_number_of_existing_models(model_path + "discriminator_epoch_")

    return max(min(counter_generator, counter_discriminator), 0)
